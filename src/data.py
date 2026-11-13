"""Official metadata audit and deterministic, resumable Fakeddit sampling."""
from __future__ import annotations

import hashlib
import html
import io
import json
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageOps

from .utils import atomic_json, file_hash, log_stage

FILES = {'train': 'multimodal_train.tsv', 'validation': 'multimodal_validate.tsv',
         'test': 'multimodal_test_public.tsv'}
SEED = 20260916
THREAD = threading.local()
# Visually verified loading spinner from official sample 4dkucm, not article imagery.
PLACEHOLDER_PIXELS = {'5c10ffff571aba003b85a447ab9828f295f7e97898e8f5e98f5661ba30aaf59e'}
SOURCES = {
    'train': ('1XsOkD3yhxgWu8URMes0S9LVSwuFJ4pT6','9894cb4aa2f901c6f29b61ac3d95ce9f7e78c43be8062ae5deb1c219c82497bb'),
    'validation': ('1Z99QrwpthioZQY2U6HElmnx8jazf7-Kv','c70140ec376e083e3780b5b488c96c8933678ca5800139bfb78c92e95df6946e'),
    'test': ('1p9EewIKVcFbipVRLZNGYc2JbSC7A0SWv','bc92378e070ca7b1a7497053d1c34cdc3f550be6232a76791a2c722d6d9bbf94'),
}


class BoundedRetry(Retry):
    def sleep_for_retry(self, response):
        delay=self.get_retry_after(response)
        if delay and delay>60:
            raise requests.exceptions.RetryError('Server Retry-After exceeds the per-image retrieval budget; no early retry attempted')
        return super().sleep_for_retry(response)


def download_metadata(root: Path):
    import gdown
    started=time.monotonic()
    directory=root/'data/raw/fakeddit'
    directory.mkdir(parents=True,exist_ok=True)
    for split,(identifier,digest) in SOURCES.items():
        path=directory/FILES[split]
        if path.exists():
            assert file_hash(path)==digest, f'Unexpected official metadata contents: {path}'
            print(f'Verified {path.name}',flush=True)
            continue
        temporary=path.with_suffix('.download')
        result=gdown.download(id=identifier,output=str(temporary),quiet=False,resume=True)
        if not result or file_hash(temporary)!=digest:
            raise RuntimeError(f'Official metadata download incomplete or source changed: {split}')
        temporary.replace(path)
    log_stage(root,'Official metadata download/verification',f'Command: `python run_experiments.py download`. Three official multimodal TSVs verified by recorded SHA-256. Runtime {time.monotonic()-started:.1f}s. Official repository: https://github.com/entitize/Fakeddit .')


def clean_text(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def image_key(url: str) -> str:
    parsed = urlsplit(html.unescape(url))
    return parsed.netloc.casefold() + parsed.path


class Groups:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        self.rank[a] += self.rank[a] == self.rank[b]


def metadata_audit(root: Path) -> pd.DataFrame:
    started = time.monotonic()
    output = root / 'data/processed'
    output.mkdir(parents=True, exist_ok=True)
    columns = ['id', 'clean_title', 'image_url', 'linked_submission_id', 'hasImage',
               '2_way_label', '6_way_label', 'subreddit', 'created_utc']
    frames, summaries = [], []
    for split, name in FILES.items():
        path = root / 'data/raw/fakeddit' / name
        frame = pd.read_csv(path, sep='\t', usecols=columns, dtype=str, keep_default_na=False)
        frame['split'] = split
        frame = frame.rename(columns={'id': 'sample_id', 'clean_title': 'text',
                                      '2_way_label': 'label', '6_way_label': 'label6'})
        frame['text'] = frame.text.map(clean_text)
        frame['text_key'] = frame.text.str.casefold()
        frame['image_key'] = frame.image_url.map(image_key)
        frame['label'] = pd.to_numeric(frame.label, errors='raise').astype(int)
        frame['label6'] = pd.to_numeric(frame.label6, errors='raise').astype(int)
        assert frame.label.isin([0, 1]).all()
        assert frame.label6.isin(range(6)).all()
        assert frame.label.eq(1).equals(frame.label6.eq(0))
        valid = frame.hasImage.str.casefold().eq('true') & frame.text.ne('') & frame.image_url.str.startswith(('https://', 'http://'))
        frame['basic_valid'] = valid
        frame['reason'] = np.where(valid, '', 'missing_text_or_image_reference')
        summaries.append({'split': split, 'metadata_rows': len(frame), 'multimodal_rows': int(valid.sum()),
                          'fake': int(frame.label.eq(0).sum()), 'real': int(frame.label.eq(1).sum()),
                          'missing_text': int(frame.text.eq('').sum()),
                          'missing_image_reference': int(frame.image_url.eq('').sum()),
                          'duplicate_text_rows': int(frame.text_key.duplicated().sum()),
                          'duplicate_image_url_rows': int(frame.loc[frame.image_key.ne(''),'image_key'].duplicated().sum()),
                          'duplicate_text_image_url_rows': int(frame[frame.image_key.ne('') & frame.text_key.ne('')].duplicated(['text_key','image_key']).sum()),
                          'metadata_sha256': file_hash(path)})
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    assert data.sample_id.str.fullmatch(r'[A-Za-z0-9]+').all(), 'Invalid sample ID'
    assert data.sample_id.is_unique, 'Official IDs overlap; investigate before sampling'
    groups = Groups(len(data))
    cross = {}
    for key in ['text_key', 'image_key']:
        cross[key] = int(data[data[key].ne('')].groupby(key).split.nunique().gt(1).sum())
    owners = {value: i for i, value in enumerate(data.sample_id)}
    linked_owners = {}
    for i, linked in enumerate(data.linked_submission_id):
        if linked:
            linked = linked.removeprefix('t3_')
            if linked in owners:
                groups.join(i, owners[linked])
            if linked in linked_owners:
                groups.join(i, linked_owners[linked])
            else:
                linked_owners[linked] = i
    data['group_id'] = [str(groups.find(i)) for i in range(len(data))]
    rank = data.split.map({'train': 0, 'validation': 1, 'test': 2})
    priority = rank.groupby(data.group_id).transform('max')
    blocked = rank.ne(priority)
    for key in ['text_key', 'image_key']:
        priority = rank.groupby(data[key]).transform('max')
        blocked |= data[key].ne('') & rank.ne(priority)
    data.loc[blocked & data.basic_valid, 'reason'] = 'cross_split_text_url_or_linked_group'
    data['eligible'] = data.reason.eq('')
    data[['sample_id','split','group_id','eligible','reason']].to_csv(output / 'metadata_audit.csv.gz', index=False)
    for summary in summaries:
        part = data[data.split.eq(summary['split'])]
        summary['cross_split_group_exclusions'] = int(part.reason.eq('cross_split_text_url_or_linked_group').sum())
        summary['eligible_metadata_rows'] = int(part.eligible.sum())
    pd.DataFrame(summaries).to_csv(output / 'metadata_summary.csv', index=False)
    atomic_json(output / 'metadata_protocol.json', {
        'dataset': 'Fakeddit v2 official multimodal_only_samples', 'seed': SEED,
        'text': 'clean_title', 'labels': {'0': 'fake', '1': 'real'},
        'source': 'https://github.com/entitize/Fakeddit', 'cross_split_duplicate_keys': cross,
        'group_rule': 'Linked-submission families form connected groups. Independently reserve each normalized-text key, image-URL key and source family to its highest original split: test > validation > train. Never reassign rows or transitively join unrelated families through generic captions.',
        'selection_rule': 'Seeded classwise candidate order; one record per group and decoded image content; quotas follow original class proportions.',
        'rows': len(data), 'runtime_seconds': time.monotonic()-started})
    print('Metadata audit:', pd.DataFrame(summaries).to_json(orient='records'), flush=True)
    log_stage(root, 'Fakeddit metadata audit', f'{len(data):,} official multimodal rows audited. Binary labels checked against all six-way labels. Cross-split connected groups identified before sampling. Runtime {time.monotonic()-started:.1f} seconds.')
    return data


def image_info(content: bytes) -> dict:
    with Image.open(io.BytesIO(content)) as opened:
        opened.verify()
    with Image.open(io.BytesIO(content)) as opened:
        opened.seek(0)
        rgb = ImageOps.exif_transpose(opened).convert('RGB')
        rgb.load()
        pixels = hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode() + rgb.tobytes()).hexdigest()
        gray = np.asarray(rgb.convert('L').resize((9, 8)))
        bits = (gray[:, 1:] > gray[:, :-1]).ravel()
        return {'sha256': hashlib.sha256(content).hexdigest(), 'pixel_sha256': pixels,
                'width': rgb.width, 'height': rgb.height, 'format': opened.format,
                'uniform_rgb': all(low==high for low,high in rgb.getextrema()),
                'bytes': len(content), 'dhash': f'{int("".join(bits.astype(int).astype(str)), 2):016x}'}


def audit_acquired(root: Path, metadata: pd.DataFrame):
    """Report overlaps in all retrieved images separately from retained-pair checks."""
    path=root/'data/processed/download_records.jsonl'
    manifest_path=root/'data/processed/manifest.csv'
    if not path.exists() or not manifest_path.exists():
        return
    records={}
    for line in path.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        records[row['sample_id']]=row
    retrieved=pd.DataFrame([r for r in records.values() if r['status']=='ok'])
    retrieved=retrieved.merge(metadata[['sample_id','split','label']],on='sample_id',validate='one_to_one')
    retained=pd.read_csv(manifest_path,keep_default_na=False)
    report={'retrieved_decodable_images':len(retrieved),'retained_pairs':len(retained),'overlap':{}}
    for name,frame in [('all_retrieved',retrieved),('retained',retained)]:
        report['overlap'][name]={}
        for key in ['sha256','pixel_sha256','original_sha256']:
            available=frame[frame[key].fillna('').ne('')]
            grouped=available.groupby(key)
            report['overlap'][name][key]={'duplicate_rows':int(available[key].duplicated().sum()),
                                         'duplicate_hash_values':int(grouped.size().gt(1).sum()),
                                         'cross_split_hash_values':int(grouped.split.nunique().gt(1).sum())}
            if name=='retained':
                assert available[key].is_unique
    atomic_json(root/'data/processed/content_overlap.json',report)


def local_cache(manifests: list[Path]) -> dict:
    cache = {}
    for manifest in manifests:
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        for row in frame.to_dict('records'):
            base = manifest.parent.parent
            cache[row['id']] = {'path': base / row['image_path'], 'url': row['image_url'],
                                'sha256': row['stored_sha256'], 'retrieved_utc': row.get('retrieved_utc', ''),
                                'original_sha256': row.get('original_sha256', '')}
    return cache


def fetch_image(row: dict, root: Path, cache: dict) -> dict:
    sid = row['sample_id']
    result = {'sample_id': sid, 'image_url': row['image_url'], 'status': 'failed', 'reason': ''}
    try:
        cached = cache.get(sid)
        content = None
        if cached and cached['url'] == row['image_url'] and cached['path'].is_file():
            content = cached['path'].read_bytes()
            if hashlib.sha256(content).hexdigest() != cached['sha256']:
                content = None
            else:
                result.update(origin='verified_local_image_cache', retrieved_utc=cached['retrieved_utc'],
                              resolved_url=row['image_url'], original_sha256=cached['original_sha256'])
        if content is None:
            if not hasattr(THREAD, 'session'):
                THREAD.session = requests.Session()
                THREAD.session.headers['User-Agent'] = 'Mozilla/5.0 (compatible; academic-dataset-retrieval/1.0)'
                retry = BoundedRetry(total=1, backoff_factor=0.3, status_forcelist=[429,500,502,503,504], allowed_methods=['GET'], raise_on_status=False)
                THREAD.session.mount('https://', HTTPAdapter(max_retries=retry))
                THREAD.session.mount('http://', HTTPAdapter(max_retries=retry))
            url = html.unescape(row['image_url'])
            if url.startswith('http://i.imgur.com/'):
                url = 'https://' + url[len('http://'):]
            transfer_started=time.monotonic()
            with THREAD.session.get(url, stream=True, timeout=(5, 18)) as response:
                if response.status_code != 200:
                    result['reason'] = f'http_{response.status_code}'
                    return result
                if any(token in response.url.lower() for token in ['/removed.', '/unavailable.', '/image_removed']):
                    result['reason'] = 'removed_image_placeholder'
                    return result
                blocks, total = [], 0
                for block in response.iter_content(65536):
                    if time.monotonic()-transfer_started>120:
                        result['reason']='transfer_over_120_seconds'
                        return result
                    total += len(block)
                    if total > 15 * 1024 * 1024:
                        result['reason'] = 'download_over_15_mib'
                        return result
                    blocks.append(block)
                content = b''.join(blocks)
                result.update(origin='official_metadata_image_url', resolved_url=response.url,
                              retrieved_utc=datetime.now(timezone.utc).isoformat(),
                              original_sha256=hashlib.sha256(content).hexdigest())
        info = image_info(content)
        result.update(info)
        ext = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif', 'BMP': '.bmp', 'TIFF': '.tif'}.get(info['format'])
        if ext is None:
            result['reason'] = 'unsupported_image_format'
            return result
        directory = root / 'data/raw/fakeddit/images'
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / (sid + ext)
        temporary = destination.with_suffix(ext + '.part')
        temporary.write_bytes(content)
        temporary.replace(destination)
        result.update(status='ok', image_path=destination.relative_to(root).as_posix())
    except requests.RequestException as exc:
        result['reason'] = type(exc).__name__
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        result['reason'] = 'image_decode_' + type(exc).__name__
    return result


def prepare(root: Path, counts: dict, workers: int = 24, cache_manifests: list[Path] | None = None) -> pd.DataFrame:
    started = time.monotonic()
    atomic_json(root/'data/processed/preparation_progress.json',{'status':'running','stage':'metadata audit','counts':counts})
    data = metadata_audit(root)
    output = root / 'data/processed'
    progress = output / 'download_records.jsonl'
    records = {}
    if progress.exists():
        lines = progress.read_text(encoding='utf-8').splitlines()
        for i, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1:
                    raise
                progress.write_text('\n'.join(lines[:i])+'\n', encoding='utf-8')
                break
            records[record['sample_id']] = record
        upgraded=False
        for record in records.values():
            if record.get('status')=='ok' and 'uniform_rgb' not in record:
                path=root/record['image_path']
                if path.is_file():
                    with Image.open(path) as opened:
                        rgb=ImageOps.exif_transpose(opened).convert('RGB')
                        record['uniform_rgb']=all(low==high for low,high in rgb.getextrema())
                        rgb.close()
                    upgraded=True
        if upgraded:
            temporary=progress.with_suffix('.part')
            temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records.values()),encoding='utf-8')
            temporary.replace(progress)
    cache = local_cache(cache_manifests or [])
    selected, decisions, seen_groups, seen_pixels, seen_files = [], [], set(), set(), set()
    seen_texts, seen_urls, seen_originals = set(), set(), set()
    with ThreadPoolExecutor(max_workers=workers) as pool, progress.open('a', encoding='utf-8') as stream:
        for split_index, split in enumerate(['test', 'validation', 'train']):
            part = data[data.split.eq(split)]
            target = counts[split]
            real_target = int(round(target * part.label.mean()))
            for label, quota in [(0, target-real_target), (1, real_target)]:
                candidates = part[part.eligible & part.label.eq(label)].copy()
                rng = np.random.default_rng(SEED + 10*split_index + label)
                candidates = candidates.iloc[rng.permutation(len(candidates))]
                accepted = 0
                queue = candidates.to_dict('records')
                offset=0
                while offset<len(queue) and accepted<quota:
                    window=min(2048,max(32,quota-accepted))
                    batch = queue[offset:offset+window]
                    offset+=len(batch)
                    need = [r for r in batch if r['group_id'] not in seen_groups and
                            (r['sample_id'] not in records or
                             (records[r['sample_id']].get('status')=='ok' and
                              not (root/records[r['sample_id']]['image_path']).is_file()))]
                    futures = [pool.submit(fetch_image,row,root,cache) for row in need]
                    for future in as_completed(futures):
                        record = future.result()
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                        stream.flush()
                        records[record['sample_id']] = record
                    for row in batch:
                        if accepted >= quota:
                            break
                        sid, reason = row['sample_id'], ''
                        record = records.get(sid, {})
                        if row['group_id'] in seen_groups:
                            reason = 'duplicate_content_group'
                        elif row['text_key'] in seen_texts:
                            reason = 'duplicate_text'
                        elif row['image_key'] in seen_urls:
                            reason = 'duplicate_image_url'
                        elif record.get('status') != 'ok':
                            reason = record.get('reason', 'not_downloaded')
                        elif record['pixel_sha256'] in PLACEHOLDER_PIXELS:
                            reason = 'visually_verified_loading_placeholder'
                        elif record.get('uniform_rgb'):
                            reason = 'uniform_blank_source_image'
                        elif record['pixel_sha256'] in seen_pixels or record['sha256'] in seen_files or record.get('original_sha256') in seen_originals:
                            reason = 'duplicate_image_content'
                        else:
                            path = root / record['image_path']
                            if not path.is_file() or file_hash(path) != record['sha256']:
                                raise ValueError(f'Cached image missing or modified: {sid}')
                        decisions.append({'sample_id': sid, 'split': split, 'label': label, 'decision': reason or 'selected'})
                        if reason:
                            continue
                        selected.append({**row, **record})
                        seen_groups.add(row['group_id'])
                        seen_texts.add(row['text_key'])
                        seen_urls.add(row['image_key'])
                        seen_pixels.add(record['pixel_sha256'])
                        seen_files.add(record['sha256'])
                        if record.get('original_sha256'):
                            seen_originals.add(record['original_sha256'])
                        accepted += 1
                    print(f'{split} class {label}: {accepted}/{quota}; selected {len(selected)}; elapsed {time.monotonic()-started:.0f}s', flush=True)
                    atomic_json(output/'preparation_progress.json',{'status':'running','split':split,'label':label,
                                'accepted_in_class':accepted,'quota':quota,'selected':len(selected),
                                'recorded_attempts':len(records),'elapsed_seconds':time.monotonic()-started})
                if accepted < quota:
                    raise RuntimeError(f'Exhausted {split} label {label}: {accepted}/{quota}; no silent reduction')
    manifest = pd.DataFrame(selected).sort_values(['split','sample_id']).reset_index(drop=True)
    assert not manifest.uniform_rgb.any(), 'A blank source image escaped selection'
    keep = ['sample_id','split','label','text','image_path','image_url','group_id','linked_submission_id',
            'label6','subreddit','created_utc','sha256','pixel_sha256','width','height','format','bytes',
            'dhash','origin','original_sha256','retrieved_utc','resolved_url']
    manifest = manifest[keep]
    assert manifest.sample_id.is_unique and manifest.group_id.is_unique
    assert manifest.sha256.is_unique and manifest.pixel_sha256.is_unique
    assert manifest.loc[manifest.original_sha256.ne(''),'original_sha256'].is_unique
    assert manifest.text.map(clean_text).str.casefold().is_unique
    manifest.to_csv(output / 'manifest.csv', index=False)
    for split in FILES:
        manifest[manifest.split.eq(split)].to_csv(output / f'{split}.csv', index=False)
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(output / 'selection_audit.csv', index=False)
    downloads=pd.DataFrame(records.values()).merge(data[['sample_id','split','label']],on='sample_id',validate='one_to_one')
    downloads.groupby(['split','label','status','reason'],dropna=False).size().rename('count').to_csv(output/'download_summary.csv')
    decision_frame.groupby(['split','label','decision']).size().rename('count').to_csv(output/'exclusion_summary.csv')
    summary = []
    for split, group in manifest.groupby('split'):
        attempted = decision_frame[decision_frame.split.eq(split)]
        summary.append({'split': split, 'samples': len(group), 'fake': int(group.label.eq(0).sum()),
                        'real': int(group.label.eq(1).sum()), 'excluded_attempts': int(attempted.decision.ne('selected').sum()),
                        'missing_physical_images': 0, 'corrupt_images': 0, 'duplicate_text': 0,
                        'duplicate_image_hashes': 0, 'duplicate_text_image_pairs': 0, 'cross_split_exact_duplicates': 0})
    pd.DataFrame(summary).to_csv(output / 'data_summary.csv', index=False)
    manifest.groupby(['split','format']).size().rename('count').to_csv(output / 'image_formats.csv')
    manifest.groupby(['split','label','label6','subreddit']).size().rename('count').to_csv(output/'source_distribution.csv')
    manifest[['width','height','bytes']].describe(percentiles=[.5,.9,.95,.99]).to_csv(output / 'image_dimensions.csv')
    manifest.text.str.split().str.len().describe(percentiles=[.5,.9,.95,.99]).to_csv(output / 'text_lengths.csv')
    atomic_json(output / 'selection.json', {'seed': SEED, 'counts': counts, 'manifest_sha256': file_hash(output / 'manifest.csv'),
                                          'runtime_seconds': time.monotonic()-started,
                                          'recorded_download_attempts': len(records), 'summary': summary})
    atomic_json(output/'preparation_progress.json',{'status':'complete','selected':len(manifest),'counts':counts,
                                                   'elapsed_seconds':time.monotonic()-started})
    log_stage(root, 'Fakeddit image preparation', f'Selected {len(manifest):,} decodable pairs: {counts}. No shared normalized text, file hash, decoded pixel hash or metadata group. Runtime {time.monotonic()-started:.1f} seconds. Exact exclusions: selection_audit.csv and metadata_audit.csv.gz. Image sources are official image URLs or matching hash-verified local files.')
    return manifest

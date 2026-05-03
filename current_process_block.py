def _process_image_entries(entries: List[Tuple[str, str]], user_agent: str) -> List[Dict[str, object]]:
    if not entries:
        return []
    filtered = []
    for url, alt in entries:
        if not url or url.lower().startswith("data:"):
            continue
        filtered.append((url, alt))
    if not filtered:
        return []

    global HTTP_WARNING_ISSUED, PIL_WARNING_ISSUED

    if not (aiohttp or requests):
        if not HTTP_WARNING_ISSUED:
            print("[images] HTTP client libraries unavailable; skipping image downloads")
            HTTP_WARNING_ISSUED = True
        return [_build_image_record(url, alt, None) for url, alt in filtered]

    results: List[Tuple[str, str, Optional[bytes]]] = []
    if aiohttp:
        try:
            results = asyncio.run(_download_images_async(filtered))
        except Exception:
            results = []
    if not results:
        results = _download_images_sync(filtered)

    if not results:
        return [_build_image_record(url, alt, None) for url, alt in filtered]

    if Image is None and not PIL_WARNING_ISSUED:
        print("[images] Pillow not available; image metadata will be limited")
        PIL_WARNING_ISSUED = True

    return [_build_image_record(url, alt, data) for url, alt, data in results]

export function getBCGUserAgent(version: string): string {
	const runtime = process.versions.bun ? `bun/${process.versions.bun}` : `node/${process.version}`;
	return `bcg/${version} (${process.platform}; ${runtime}; ${process.arch})`;
}

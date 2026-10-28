import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash, randomFillSync} from 'node:crypto';
import {randomId, sha256Hex} from '../browser_crypto.mjs';

const insecure = {getRandomValues: bytes => randomFillSync(bytes)};
const expected = bytes => createHash('sha256').update(bytes).digest('hex');

test('plain HTTP UUID fallback retains v4 format and uniqueness', () => {
  const ids = new Set(Array.from({length: 1000}, () => randomId(insecure)));
  assert.equal(ids.size, 1000);
  for (const id of ids) assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
});

test('uses native randomUUID when available', () => {
  assert.equal(randomId({randomUUID: () => 'native-id'}), 'native-id');
});

test('native and plain HTTP SHA-256 agree with standard vectors', async () => {
  for (const text of ['', 'abc', 'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq', '证据来源与置信度', 'a'.repeat(1_000_000)]) {
    const bytes = new TextEncoder().encode(text);
    assert.equal(await sha256Hex(bytes, insecure), expected(bytes));
    assert.equal(await sha256Hex(bytes), expected(bytes));
    assert.equal(await sha256Hex(bytes.buffer, insecure), expected(bytes));
  }
});

test('fallback handles padding boundaries, binary data, and buffer offsets', async () => {
  for (const size of [1, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 4096]) {
    const bytes = randomFillSync(new Uint8Array(size + 12)).subarray(5, size + 5);
    assert.equal(await sha256Hex(bytes, insecure), expected(bytes));
    assert.equal(await sha256Hex(new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), insecure), expected(bytes));
  }
});

test('large-file fallback yields while computing the same full-file hash', async () => {
  const bytes = new Uint8Array(5 * 1024 * 1024 + 57).fill(0xa5);
  let yielded = false;
  setTimeout(() => { yielded = true; }, 0);
  assert.equal(await sha256Hex(bytes, insecure), expected(bytes));
  assert.equal(yielded, true);
});

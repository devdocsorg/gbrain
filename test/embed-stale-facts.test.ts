/**
 * Tests for embedStaleFacts (src/core/embed-stale.ts).
 *
 * Hermetic — an injected `embedFn` means no network call lands. Validates the
 * drain that refills `facts.embedding` after a dimension transition NULLed it:
 *   - no NULL facts → done with zero work
 *   - every active NULL fact is embedded at the column width; expired facts
 *     and already-embedded facts are untouched
 *   - maxFacts bounds one call and the next call resumes (NULL is the checkpoint)
 *   - a wrong-width vector is counted as failed, never written, never thrown
 *   - an embedder throw leaves the rows NULL and reports them as failed
 *   - a pre-aborted signal returns aborted without touching rows
 *   - after runSchemaTransition to a new width, the drain refills at that width
 */
import { describe, test, expect, beforeAll, afterAll, beforeEach } from 'bun:test';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { resetPgliteState } from './helpers/reset-pglite.ts';
import { embedStaleFacts } from '../src/core/embed-stale.ts';
import { readFactsEmbeddingDim } from '../src/core/embedding-dim-check.ts';
import { runSchemaTransition } from '../src/core/embedding-migration.ts';

let engine: PGLiteEngine;

beforeAll(async () => {
  engine = new PGLiteEngine();
  await engine.connect({});
  await engine.initSchema();
}, 30000);

afterAll(async () => {
  await engine.disconnect();
});

beforeEach(async () => {
  await resetPgliteState(engine);
});

async function factsWidth(): Promise<number> {
  const col = await readFactsEmbeddingDim(engine);
  if (!col.exists || col.dims === null) throw new Error('facts.embedding column missing');
  return col.dims;
}

function fakeEmbedFn(width: number) {
  const calls: string[][] = [];
  const fn = async (texts: string[]): Promise<Float32Array[]> => {
    calls.push(texts);
    return texts.map((text) => {
      const v = new Float32Array(width);
      v[0] = 1;
      v[1] = text.length / 1000;
      return v;
    });
  };
  return { fn, calls };
}

async function seedFact(fact: string, opts: { embedding?: Float32Array | null; expired?: boolean } = {}): Promise<number> {
  const { id } = await engine.insertFact(
    {
      fact,
      source: 'test:embed-stale-facts',
      embedding: opts.embedding ?? null,
    },
    { source_id: 'default' },
  );
  if (opts.expired) {
    await engine.executeRaw(`UPDATE facts SET expired_at = now() WHERE id = $1`, [id]);
  }
  return id;
}

async function nullFactCount(): Promise<number> {
  const rows = await engine.executeRaw<{ n: number }>(
    `SELECT count(*)::int AS n FROM facts WHERE embedding IS NULL AND expired_at IS NULL`,
  );
  return Number(rows[0]?.n ?? 0);
}

describe('embedStaleFacts', () => {
  test('returns done with zero work when no active fact is missing an embedding', async () => {
    const width = await factsWidth();
    const { fn, calls } = fakeEmbedFn(width);
    expect(await embedStaleFacts(engine, { embedFn: fn })).toEqual({
      embedded: 0,
      failed: 0,
      done: true,
      aborted: false,
    });
    expect(calls).toEqual([]);
  });

  test('embeds every active NULL fact and leaves expired and embedded facts alone', async () => {
    const width = await factsWidth();
    const existing = new Float32Array(width);
    existing[0] = 0.5;
    await seedFact('stale fact one');
    await seedFact('stale fact two');
    await seedFact('already embedded fact', { embedding: existing });
    await seedFact('expired stale fact', { expired: true });
    const { fn, calls } = fakeEmbedFn(width);

    const result = await embedStaleFacts(engine, { embedFn: fn, batchSize: 1 });

    expect(result).toEqual({ embedded: 2, failed: 0, done: true, aborted: false });
    expect(calls.flat().sort()).toEqual(['stale fact one', 'stale fact two']);
    expect(await nullFactCount()).toBe(0);
    const expired = await engine.executeRaw<{ has: boolean }>(
      `SELECT embedding IS NOT NULL AS has FROM facts WHERE fact = 'expired stale fact'`,
    );
    expect(expired[0]?.has).toBe(false);
  });

  test('maxFacts bounds one call and the next call resumes from the NULL checkpoint', async () => {
    const width = await factsWidth();
    for (let i = 0; i < 5; i++) await seedFact(`bounded fact ${i}`);
    const { fn } = fakeEmbedFn(width);

    const first = await embedStaleFacts(engine, { embedFn: fn, batchSize: 2, maxFacts: 3 });
    expect(first.embedded).toBe(3);
    expect(first.done).toBe(false);
    expect(await nullFactCount()).toBe(2);

    const second = await embedStaleFacts(engine, { embedFn: fn, batchSize: 2 });
    expect(second).toEqual({ embedded: 2, failed: 0, done: true, aborted: false });
    expect(await nullFactCount()).toBe(0);
  });

  test('counts wrong-width vectors as failed without writing them', async () => {
    const width = await factsWidth();
    await seedFact('wrong width fact');
    const { fn } = fakeEmbedFn(width + 1);

    const result = await embedStaleFacts(engine, { embedFn: fn });

    expect(result).toEqual({ embedded: 0, failed: 1, done: true, aborted: false });
    expect(await nullFactCount()).toBe(1);
  });

  test('an embedder throw leaves the rows NULL and reports them failed', async () => {
    await seedFact('provider outage fact');
    const result = await embedStaleFacts(engine, {
      embedFn: async () => {
        throw new Error('provider unavailable');
      },
    });
    expect(result).toEqual({ embedded: 0, failed: 1, done: true, aborted: false });
    expect(await nullFactCount()).toBe(1);
  });

  test('a pre-aborted signal returns aborted without touching rows', async () => {
    const width = await factsWidth();
    await seedFact('aborted fact');
    const controller = new AbortController();
    controller.abort();
    const { fn, calls } = fakeEmbedFn(width);

    const result = await embedStaleFacts(engine, { embedFn: fn, signal: controller.signal });

    expect(result).toEqual({ embedded: 0, failed: 0, done: false, aborted: true });
    expect(calls).toEqual([]);
    expect(await nullFactCount()).toBe(1);
  });

  test('refills facts at the new width after a dimension transition', async () => {
    const width = await factsWidth();
    const original = new Float32Array(width);
    original[0] = 1;
    await seedFact('fact that survives a transition', { embedding: original });
    const target = width === 768 ? 512 : 768;

    await runSchemaTransition(engine, target);
    expect((await readFactsEmbeddingDim(engine)).dims).toBe(target);
    expect(await nullFactCount()).toBe(1);

    const { fn } = fakeEmbedFn(target);
    const result = await embedStaleFacts(engine, { embedFn: fn });

    expect(result).toEqual({ embedded: 1, failed: 0, done: true, aborted: false });
    expect(await nullFactCount()).toBe(0);
    const kept = await engine.executeRaw<{ fact: string }>(`SELECT fact FROM facts`);
    expect(kept.map((row) => row.fact)).toEqual(['fact that survives a transition']);
  });
});

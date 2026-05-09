/**
 * Tests for the synchronous MSM API exposed on `Curve.Sync` (Pasta only).
 *
 * Verifies:
 * - sync return type (Uint8Array, no Promise)
 * - parity vs `Bigint.Projective.msm` for Pallas + Vesta
 * - input validation
 * - basic timing smoke for 2^10 / 2^14
 *
 * Runs without `startThreads` — the sync API must work on a single thread.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { Pallas, Vesta, startThreads, stopThreads } from "./index.ts";
import { type CurveParams } from "./bigint/affine-weierstrass.ts";

type Weierstraß = Awaited<ReturnType<typeof Pallas>>;

let pallas = await Pallas();
let vesta = await Vesta();

test("Pallas Sync.msmAffineBytes parity vs Bigint.Projective.msm", async () => {
  for (let N of [0, 1, 2, 3, 8, 1024]) {
    await checkParity(pallas, N);
  }
});

test("Vesta Sync.msmAffineBytes parity vs Bigint.Projective.msm", async () => {
  for (let N of [0, 1, 2, 3, 8, 1024]) {
    await checkParity(vesta, N);
  }
});

test("Sync.msmAffineBytes returns Uint8Array, not a Promise", () => {
  let out = pallas.Sync.msmAffineBytes(new Uint8Array(0), new Uint8Array(0));
  assert.ok(out instanceof Uint8Array, "expected Uint8Array");
  assert.equal(out.length, 64);
  assert.equal(typeof (out as any).then, "undefined", "must not be a Promise");
});

test("Sync.msmAffineBytes rejects malformed input", () => {
  // points length not a multiple of 64
  assert.throws(
    () => pallas.Sync.msmAffineBytes(new Uint8Array(63), new Uint8Array(32)),
    /pointsBytes\.length divisible by 64/,
  );
  // scalar length not a multiple of 32
  assert.throws(
    () => pallas.Sync.msmAffineBytes(new Uint8Array(64), new Uint8Array(31)),
    /scalarBytes\.length divisible by 32/,
  );
  // count mismatch (2 points, 1 scalar)
  assert.throws(
    () => pallas.Sync.msmAffineBytes(new Uint8Array(128), new Uint8Array(32)),
    /point\/scalar count mismatch/,
  );
});

test("Sync vs Parallel (async) parity for Pallas + Vesta", async () => {
  await startThreads(2);
  try {
    for (let C of [pallas, vesta]) {
      for (let N of [0, 1, 2, 8, 1024]) {
        let { pointsBytes, scalarBytes } = randomInputBytes(C, N);
        let syncOut = C.Sync.msmAffineBytes(pointsBytes, scalarBytes);
        let asyncOut = await asyncMsmBytes(C, pointsBytes, scalarBytes);
        assert.deepEqual(
          [...syncOut],
          [...asyncOut],
          `${C.params.label} N=${N}: sync output != async output`,
        );
      }
    }
  } finally {
    await stopThreads();
  }
});

test("perf smoke: 2^10 and 2^14 (Pallas)", async () => {
  for (let n of [10, 14]) {
    let N = 1 << n;
    let { pointsBytes, scalarBytes } = randomInputBytes(pallas, N);

    // warmup
    pallas.Sync.msmAffineBytes(pointsBytes, scalarBytes);

    let t0 = performance.now();
    let runs = 3;
    for (let i = 0; i < runs; i++) {
      pallas.Sync.msmAffineBytes(pointsBytes, scalarBytes);
    }
    let syncMs = (performance.now() - t0) / runs;

    // Bigint baseline only at the smaller size — 2^14 is too slow
    let bigintMs: number | null = null;
    if (n <= 10) {
      let { points, scalars } = decodeForBigint(pallas, pointsBytes, scalarBytes);
      let t1 = performance.now();
      pallas.Bigint.Projective.msm(scalars, points);
      bigintMs = performance.now() - t1;
    }

    console.log(
      `[perf] Pallas N=2^${n}  sync=${syncMs.toFixed(1)}ms` +
        (bigintMs !== null
          ? `  bigint=${bigintMs.toFixed(1)}ms  speedup=${(bigintMs / syncMs).toFixed(1)}x`
          : ""),
    );
  }
});

// helpers

async function checkParity(C: Weierstraß, N: number) {
  let { pointsBytes, scalarBytes, points, scalars } = randomInputs(C, N);

  let outBytes = C.Sync.msmAffineBytes(pointsBytes, scalarBytes);
  assert.ok(outBytes instanceof Uint8Array);
  assert.equal(outBytes.length, 64);

  let expectedProj = C.Bigint.Projective.msm(scalars, points);
  let expected = C.Bigint.Projective.toAffine(expectedProj);

  if (expected.isZero) {
    for (let i = 0; i < outBytes.length; i++) {
      assert.equal(outBytes[i], 0, `N=${N}: expected all-zero infinity output`);
    }
    return;
  }

  let xLE = outBytes.slice(0, 32);
  let yLE = outBytes.slice(32, 64);
  let x = bytesToBigintLE(xLE);
  let y = bytesToBigintLE(yLE);

  assert.equal(x, expected.x, `N=${N}: x mismatch`);
  assert.equal(y, expected.y, `N=${N}: y mismatch`);
}

function randomInputs(C: Weierstraß, N: number) {
  let { pointsBytes, scalarBytes } = randomInputBytes(C, N);
  let { points, scalars } = decodeForBigint(C, pointsBytes, scalarBytes);
  return { pointsBytes, scalarBytes, points, scalars };
}

function randomInputBytes(C: Weierstraß, N: number) {
  let pointsBytes = new Uint8Array(64 * N);
  let scalarBytes = new Uint8Array(32 * N);

  let q = C.params.order;

  // generate N points by scaling the generator with random scalars in Fq
  let G = C.Bigint.Projective.fromAffine({
    x: C.params.generator.x,
    y: C.params.generator.y,
    isZero: false,
  });
  for (let i = 0; i < N; i++) {
    let kP = randomScalarBigint(q);
    let P = C.Bigint.Projective.scale(kP, G);
    let aff = C.Bigint.Projective.toAffine(P);
    bigintToBytesLE(aff.x, pointsBytes, 64 * i, 32);
    bigintToBytesLE(aff.y, pointsBytes, 64 * i + 32, 32);

    let s = randomScalarBigint(q);
    bigintToBytesLE(s, scalarBytes, 32 * i, 32);
  }

  return { pointsBytes, scalarBytes };
}

function decodeForBigint(
  C: Weierstraß,
  pointsBytes: Uint8Array,
  scalarBytes: Uint8Array,
) {
  let N = pointsBytes.length / 64;
  let points = Array(N);
  let scalars: bigint[] = Array(N);
  for (let i = 0; i < N; i++) {
    let x = bytesToBigintLE(pointsBytes.subarray(64 * i, 64 * i + 32));
    let y = bytesToBigintLE(pointsBytes.subarray(64 * i + 32, 64 * i + 64));
    points[i] = C.Bigint.Projective.fromAffine({ x, y, isZero: false });
    scalars[i] = bytesToBigintLE(
      scalarBytes.subarray(32 * i, 32 * i + 32),
    );
  }
  return { points, scalars };
}

async function asyncMsmBytes(
  C: Weierstraß,
  pointsBytes: Uint8Array,
  scalarBytes: Uint8Array,
): Promise<Uint8Array> {
  let { Field, Scalar, Affine, Projective, Parallel, Bigint } = C;
  let bytesPerPoint = 2 * Field.packedSizeField;
  let packedScalar = Scalar.packedSizeField;
  let N = pointsBytes.length / bytesPerPoint;

  let out = new Uint8Array(bytesPerPoint);
  if (N === 0) return out;

  // copy bytes into wasm and decode (mirroring Sync.msmAffineBytes)
  let pointsInputPtr = await Parallel.getPointer(N * bytesPerPoint);
  Field.memoryBytes.set(pointsBytes, pointsInputPtr);
  let scalarsInputPtr = await Parallel.getScalarPointer(N * packedScalar);
  Scalar.memoryBytes.set(scalarBytes, scalarsInputPtr);

  let pointPtr = await Parallel.getPointer(N * Affine.size);
  await Parallel.pointsFromBytes(pointPtr, pointsInputPtr, N);
  let scalarPtr = await Parallel.getScalarPointer(N * Scalar.sizeField);
  await Parallel.scalarsFromBytes(scalarPtr, scalarsInputPtr, N);

  // run async parallel msm
  let { result } = await Parallel.msm(scalarPtr, pointPtr, N);

  // pack result to LE bytes via Bigint helpers (no shared mem with the wasm
  // pointer slots used above)
  let projBigint = Projective.toBigint(result);
  let aff = Bigint.Projective.toAffine(projBigint);
  if (aff.isZero) return out;
  bigintToBytesLE(aff.x, out, 0, Field.packedSizeField);
  bigintToBytesLE(aff.y, out, Field.packedSizeField, Field.packedSizeField);
  return out;
}

function bigintToBytesLE(x: bigint, out: Uint8Array, off: number, len: number) {
  for (let i = 0; i < len; i++) {
    out[off + i] = Number(x & 0xffn);
    x >>= 8n;
  }
}

function bytesToBigintLE(b: Uint8Array): bigint {
  let x = 0n;
  for (let i = b.length - 1; i >= 0; i--) {
    x = (x << 8n) | BigInt(b[i]);
  }
  return x;
}

function randomScalarBigint(q: bigint): bigint {
  let bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  let x = bytesToBigintLE(bytes);
  return x % q;
}

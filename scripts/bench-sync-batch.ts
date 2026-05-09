/**
 * Benchmark: repeated `Sync.msmAffineBytes` calls vs one
 * `Sync.msmAffineBytesBatch` call, for the o1js prover MSM shape.
 *
 * Run: `node scripts/bench-sync-batch.ts`
 */
import { Pallas, Vesta } from "../src/index.ts";

type Curve = Awaited<ReturnType<typeof Pallas>>;

let pallas = await Pallas();
let vesta = await Vesta();

bench("Pallas 30x4096", pallas, repeat(30, 4096));
bench("Pallas 30x256", pallas, repeat(30, 256));
bench("Pallas 30x4096 + 30x256", pallas, [...repeat(30, 4096), ...repeat(30, 256)]);
bench("Vesta 30x256", vesta, repeat(30, 256));

function bench(label: string, C: Curve, sizes: number[]) {
  let { pointsBytes, scalarBytes, perBatch } = makeInputs(C, sizes);

  // warmup
  C.Sync.msmAffineBytesBatch(pointsBytes, scalarBytes, sizes);
  for (let { pts, scs } of perBatch) C.Sync.msmAffineBytes(pts, scs);

  // batch
  let t0 = performance.now();
  let batchOut = C.Sync.msmAffineBytesBatch(pointsBytes, scalarBytes, sizes);
  let batchMs = performance.now() - t0;

  // repeated single calls
  let t1 = performance.now();
  let repeatedOut = new Uint8Array(sizes.length * 64);
  for (let i = 0; i < perBatch.length; i++) {
    let { pts, scs } = perBatch[i];
    let one = C.Sync.msmAffineBytes(pts, scs);
    repeatedOut.set(one, i * 64);
  }
  let repeatedMs = performance.now() - t1;

  // sanity check parity (cheap, makes sure the bench measures equivalent work)
  let mismatch = false;
  for (let i = 0; i < batchOut.length; i++) {
    if (batchOut[i] !== repeatedOut[i]) {
      mismatch = true;
      break;
    }
  }

  console.log(
    `[${label}]  batch=${batchMs.toFixed(1)}ms  repeated=${repeatedMs.toFixed(1)}ms  ` +
      `speedup=${(repeatedMs / batchMs).toFixed(2)}x  ` +
      `parity=${mismatch ? "MISMATCH" : "ok"}`,
  );
}

function repeat(n: number, size: number): number[] {
  return Array(n).fill(size);
}

function makeInputs(C: Curve, sizes: number[]) {
  let G = C.Bigint.Projective.fromAffine({
    x: C.params.generator.x,
    y: C.params.generator.y,
    isZero: false,
  });
  let q = C.params.order;

  let totalN = sizes.reduce((a, b) => a + b, 0);
  let pointsBytes = new Uint8Array(totalN * 64);
  let scalarBytes = new Uint8Array(totalN * 32);
  let perBatch: { pts: Uint8Array; scs: Uint8Array }[] = [];

  let pOff = 0;
  let sOff = 0;
  for (let n of sizes) {
    let pts = new Uint8Array(n * 64);
    let scs = new Uint8Array(n * 32);
    for (let i = 0; i < n; i++) {
      let kP = randScalar(q);
      let P = C.Bigint.Projective.scale(kP, G);
      let aff = C.Bigint.Projective.toAffine(P);
      packLE(aff.x, pts, 64 * i, 32);
      packLE(aff.y, pts, 64 * i + 32, 32);
      packLE(randScalar(q), scs, 32 * i, 32);
    }
    pointsBytes.set(pts, pOff);
    scalarBytes.set(scs, sOff);
    perBatch.push({ pts, scs });
    pOff += n * 64;
    sOff += n * 32;
  }
  return { pointsBytes, scalarBytes, perBatch };
}

function packLE(x: bigint, out: Uint8Array, off: number, len: number) {
  for (let i = 0; i < len; i++) {
    out[off + i] = Number(x & 0xffn);
    x >>= 8n;
  }
}

function randScalar(q: bigint): bigint {
  let bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  let x = 0n;
  for (let i = bytes.length - 1; i >= 0; i--) x = (x << 8n) | BigInt(bytes[i]);
  return x % q;
}

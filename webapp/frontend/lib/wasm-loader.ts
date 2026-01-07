// WASM loader utility for Next.js
// This file helps load the WASM module correctly in Next.js

let wasmModule: any = null;
let initPromise: Promise<any> | null = null;

export async function loadWasmModule() {
  if (wasmModule) {
    return wasmModule;
  }

  if (initPromise) {
    return initPromise;
  }

  initPromise = (async () => {
    // Fetch the WASM binary
    const wasmBinaryResponse = await fetch('/wasm/diffusion_wasm_bg.wasm');
    if (!wasmBinaryResponse.ok) {
      throw new Error('Failed to fetch WASM binary');
    }
    const wasmBinary = await wasmBinaryResponse.arrayBuffer();

    // Fetch and evaluate the JS glue code
    const jsResponse = await fetch('/wasm/diffusion_wasm.js');
    if (!jsResponse.ok) {
      throw new Error('Failed to fetch WASM JS glue');
    }
    const jsCode = await jsResponse.text();

    // Create a blob URL for the module
    const blob = new Blob([jsCode], { type: 'application/javascript' });
    const blobUrl = URL.createObjectURL(blob);

    // Import the module
    const module = await import(/* webpackIgnore: true */ blobUrl);
    
    // Initialize with the WASM binary
    await module.default(wasmBinary);
    
    // Clean up
    URL.revokeObjectURL(blobUrl);
    
    wasmModule = module;
    return module;
  })();

  return initPromise;
}

export function getDiffusionModel() {
  if (!wasmModule) {
    throw new Error('WASM module not loaded. Call loadWasmModule() first.');
  }
  return wasmModule.DiffusionModel;
}

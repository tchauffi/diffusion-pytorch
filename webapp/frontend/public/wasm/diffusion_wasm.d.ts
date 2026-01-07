/* tslint:disable */
/* eslint-disable */

export class DiffusionModel {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Load the model from SafeTensors bytes
   *
   * # Arguments
   * * `vae_weights` - VAE decoder weights as SafeTensors bytes
   * * `unet_weights` - UNet weights as SafeTensors bytes
   * * `num_classes` - Number of classes for conditional generation
   */
  constructor(vae_weights: Uint8Array, unet_weights: Uint8Array, num_classes: number);
  /**
   * Generate an image from noise
   *
   * # Arguments
   * * `seed` - Random seed for reproducibility
   * * `num_steps` - Number of diffusion steps (fewer = faster but lower quality)
   * * `class_label` - Class label for conditional generation (0-2 for cat/dog/wild)
   * * `cfg_scale` - Classifier-free guidance scale (1.0 = no guidance, higher = stronger)
   *
   * # Returns
   * RGBA image data as a flat Uint8Array (128*128*4 bytes)
   */
  generate(seed: bigint, num_steps: number, class_label: number, cfg_scale: number): Uint8Array;
  /**
   * Get the expected image dimensions
   */
  readonly image_size: number;
  /**
   * Get the number of classes
   */
  readonly num_classes: number;
}

/**
 * Initialize panic hook for better error messages in browser console
 */
export function init(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_diffusionmodel_free: (a: number, b: number) => void;
  readonly diffusionmodel_generate: (a: number, b: bigint, c: number, d: number, e: number) => [number, number, number, number];
  readonly diffusionmodel_image_size: (a: number) => number;
  readonly diffusionmodel_new: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
  readonly diffusionmodel_num_classes: (a: number) => number;
  readonly init: () => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

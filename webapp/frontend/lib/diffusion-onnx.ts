/**
 * ONNX Runtime Web-based diffusion inference
 * Uses offset cosine schedule matching the Python training code
 */
import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime (client-side only)
if (typeof window !== 'undefined') {
  // Enable WebGPU if available
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;
}

export interface DiffusionConfig {
  numClasses: number;
  latentChannels: number;
  latentSize: number;  // 16 for 128x128 images
  numSteps: number;    // Training steps (1000)
  imageSize: number;   // Output image size (128)
}

const DEFAULT_CONFIG: DiffusionConfig = {
  numClasses: 3,
  latentChannels: 4,
  latentSize: 16,
  numSteps: 1000,
  imageSize: 128,
};

// VAE scale factor (must match training)
const VAE_SCALE_FACTOR = 0.38;

/**
 * Offset cosine schedule matching Python implementation
 * Returns [noiseRate, signalRate] for a normalized diffusion time
 */
function offsetCosineSchedule(t: number): [number, number] {
  const minSignalRate = 0.02;
  const maxSignalRate = 0.95;
  
  const startAngle = Math.acos(maxSignalRate);
  const endAngle = Math.acos(minSignalRate);
  
  const angle = startAngle + (endAngle - startAngle) * t;
  
  const signalRate = Math.cos(angle);  // beta in Python code
  const noiseRate = Math.sin(angle);   // alpha in Python code
  
  return [noiseRate, signalRate];
}

export class DiffusionInference {
  private unetSession: ort.InferenceSession | null = null;
  private vaeSession: ort.InferenceSession | null = null;
  private config: DiffusionConfig;

  constructor(config: Partial<DiffusionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async loadModels(
    unetPath: string,
    vaePath: string,
    onProgress?: (message: string) => void
  ): Promise<void> {
    onProgress?.('Loading UNet model...');
    
    // Try WebGPU first, fallback to WASM
    // WebGPU provides GPU acceleration in supported browsers
    const executionProviders: ort.InferenceSession.ExecutionProviderConfig[] = ['webgpu', 'wasm'];
    
    // Load UNet
    try {
      this.unetSession = await ort.InferenceSession.create(unetPath, {
        executionProviders,
        graphOptimizationLevel: 'all',
      });
      onProgress?.(`UNet loaded (using ${this.unetSession.handler?.['_ep'] || 'unknown'} backend)`);
    } catch (e) {
      console.warn('WebGPU not available, falling back to WASM', e);
      this.unetSession = await ort.InferenceSession.create(unetPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      onProgress?.('UNet loaded (using WASM backend)');
    }
    
    onProgress?.('Loading VAE decoder...');
    
    // Load VAE decoder
    try {
      this.vaeSession = await ort.InferenceSession.create(vaePath, {
        executionProviders,
        graphOptimizationLevel: 'all',
      });
    } catch (e) {
      this.vaeSession = await ort.InferenceSession.create(vaePath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }
    
    onProgress?.('Models loaded successfully!');
  }

  async generate(
    classLabel: number,
    numSteps: number = 50,
    cfgScale: number = 3.0,
    seed?: number,
    onProgress?: (step: number, totalSteps: number) => void
  ): Promise<ImageData> {
    if (!this.unetSession || !this.vaeSession) {
      throw new Error('Models not loaded');
    }

    const { latentChannels, latentSize, numSteps: totalSteps, numClasses } = this.config;
    
    // Initialize random latent from pure noise
    let z = this.randomNormal([1, latentChannels, latentSize, latentSize], seed);
    
    // Create timestep subsequence (from high noise to low noise)
    // Same logic as Python: timesteps = (torch.arange(num_inference_steps) * step_ratio).long()
    const stepRatio = totalSteps / numSteps;
    const timesteps: number[] = [];
    for (let i = 0; i < numSteps; i++) {
      timesteps.push(Math.floor(i * stepRatio));
    }
    timesteps.reverse(); // From high to low (reverse order)
    
    // DDIM sampling loop
    for (let i = 0; i < timesteps.length; i++) {
      const t = timesteps[i];
      const tNorm = t / totalSteps;  // Normalized time [0, 1]
      
      // Call progress callback and yield to allow UI updates
      onProgress?.(i + 1, timesteps.length);
      await new Promise(resolve => setTimeout(resolve, 0));  // Yield to event loop
      
      // Get noise and signal rates for current timestep
      const [noiseRate, signalRate] = offsetCosineSchedule(tNorm);
      
      // Predict noise with classifier-free guidance
      const noisePred = await this.predictNoiseCFG(z, tNorm, classLabel, cfgScale, numClasses);
      
      // DDIM: predict z0 from zt and predicted noise
      // z_t = signalRate * z_0 + noiseRate * noise
      // z_0 = (z_t - noiseRate * noisePred) / signalRate
      const z0Pred = new Float32Array(z.length);
      for (let j = 0; j < z.length; j++) {
        const z0 = (z[j] - noiseRate * noisePred[j]) / signalRate;
        // Clamp z0 to [-3, 3] like Python does
        z0Pred[j] = Math.max(-3, Math.min(3, z0));
      }
      
      if (i < timesteps.length - 1) {
        // Get previous timestep rates
        const tPrev = timesteps[i + 1];
        const tPrevNorm = tPrev / totalSteps;
        const [noiseRatePrev, signalRatePrev] = offsetCosineSchedule(tPrevNorm);
        
        // DDIM deterministic update
        // z_{t-1} = signalRatePrev * z0_pred + noiseRatePrev * noisePred
        for (let j = 0; j < z.length; j++) {
          z[j] = signalRatePrev * z0Pred[j] + noiseRatePrev * noisePred[j];
        }
      } else {
        // Final step: use predicted clean latent
        z = z0Pred;
      }
    }
    
    // Decode latent to image
    const image = await this.decode(z);
    
    return this.tensorToImageData(image);
  }

  private async predictNoiseCFG(
    z: Float32Array,
    tNorm: number,
    classLabel: number,
    cfgScale: number,
    numClasses: number
  ): Promise<Float32Array> {
    const { latentChannels, latentSize } = this.config;
    
    // Prepare inputs
    const latentTensor = new ort.Tensor('float32', z, [1, latentChannels, latentSize, latentSize]);
    // Pass normalized timestep (matching Python: t_input / self.num_steps)
    const timestepTensor = new ort.Tensor('float32', new Float32Array([tNorm]), [1]);
    
    // Conditional prediction
    const classLabelTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(classLabel)]), [1]);
    const condResult = await this.unetSession!.run({
      latent: latentTensor,
      timestep: timestepTensor,
      class_label: classLabelTensor,
    });
    const condNoise = condResult.noise_pred.data as Float32Array;
    
    // If CFG scale is 1, no need for unconditional
    if (cfgScale <= 1.0) {
      return condNoise;
    }
    
    // Unconditional prediction (null class = numClasses)
    const nullClassTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(numClasses)]), [1]);
    const uncondResult = await this.unetSession!.run({
      latent: latentTensor,
      timestep: timestepTensor,
      class_label: nullClassTensor,
    });
    const uncondNoise = uncondResult.noise_pred.data as Float32Array;
    
    // CFG: uncond + scale * (cond - uncond)
    const guidedNoise = new Float32Array(condNoise.length);
    for (let i = 0; i < condNoise.length; i++) {
      guidedNoise[i] = uncondNoise[i] + cfgScale * (condNoise[i] - uncondNoise[i]);
    }
    
    return guidedNoise;
  }

  private async decode(latent: Float32Array): Promise<Float32Array> {
    const { latentChannels, latentSize } = this.config;
    
    // Apply VAE scale factor (divide by scale_factor like Python)
    const scaledLatent = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i++) {
      scaledLatent[i] = latent[i] / VAE_SCALE_FACTOR;
    }
    
    const latentTensor = new ort.Tensor('float32', scaledLatent, [1, latentChannels, latentSize, latentSize]);
    
    const result = await this.vaeSession!.run({ latent: latentTensor });
    return result.image.data as Float32Array;
  }

  private randomNormal(shape: number[], seed?: number): Float32Array {
    const size = shape.reduce((a, b) => a * b, 1);
    const result = new Float32Array(size);
    
    // Seeded random using Box-Muller transform
    let state = seed ?? Math.floor(Math.random() * 2147483647);
    const random = () => {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 2147483647;
    };
    
    for (let i = 0; i < size; i += 2) {
      const u1 = Math.max(random(), 1e-10);  // Avoid log(0)
      const u2 = random();
      const r = Math.sqrt(-2 * Math.log(u1));
      result[i] = r * Math.cos(2 * Math.PI * u2);
      if (i + 1 < size) {
        result[i + 1] = r * Math.sin(2 * Math.PI * u2);
      }
    }
    
    return result;
  }

  private tensorToImageData(tensor: Float32Array): ImageData {
    // Tensor is [1, 3, H, W] in approximately [-1, 1] range
    const size = this.config.imageSize;
    const imageData = new ImageData(size, size);
    
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const pixelIdx = (y * size + x) * 4;
        for (let c = 0; c < 3; c++) {
          const tensorIdx = c * size * size + y * size + x;
          // Denormalize from [-1, 1] to [0, 1], then clamp, then scale to [0, 255]
          // Matches Python: images = (images + 1) / 2; images = torch.clamp(images, 0, 1)
          const normalized = (tensor[tensorIdx] + 1) * 0.5;
          const clamped = Math.max(0, Math.min(1, normalized));
          imageData.data[pixelIdx + c] = Math.round(clamped * 255);
        }
        imageData.data[pixelIdx + 3] = 255; // Alpha
      }
    }
    
    return imageData;
  }
}

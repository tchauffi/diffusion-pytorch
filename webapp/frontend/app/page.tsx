'use client';

import { useState, useEffect, useCallback } from 'react';
import { DiffusionInference } from '@/lib/diffusion-onnx';
import { makeAssetPath } from '@/lib/asset-path';

// Class options for the model
const CLASS_OPTIONS = [
  { id: 0, name: '🐱 Cat', emoji: '🐱' },
  { id: 1, name: '🐕 Dog', emoji: '🐕' },
  { id: 2, name: '🦁 Wild', emoji: '🦁' },
];

export default function Home() {
  const resolveModelPath = (relative: string) => makeAssetPath(relative);

  const [model, setModel] = useState<DiffusionInference | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingStatus, setLoadingStatus] = useState('Initializing...');
  const [generating, setGenerating] = useState(false);
  const [generationStep, setGenerationStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  // Generation parameters
  const [seed, setSeed] = useState(42);
  const [numSteps, setNumSteps] = useState(20);
  const [classLabel, setClassLabel] = useState(0);
  const [cfgScale, setCfgScale] = useState(3.0);
  const [numImages, setNumImages] = useState(4);
  
  // Generated images
  const [generatedImages, setGeneratedImages] = useState<Array<{
    imageData: ImageData;
    seed: number;
    classLabel: number;
  }>>([]);

  // Load the ONNX models
  useEffect(() => {
    async function loadModels() {
      try {
        setLoadingStatus('Initializing ONNX Runtime...');
        setLoadingProgress(10);
        
        const diffusion = new DiffusionInference({ numClasses: 3 });
        
        setLoadingProgress(20);
        
        await diffusion.loadModels(
          resolveModelPath('/models/unet.onnx'),
          resolveModelPath('/models/vae-decoder.onnx'),
          (message) => {
            setLoadingStatus(message);
            if (message.includes('UNet')) setLoadingProgress(40);
            if (message.includes('VAE')) setLoadingProgress(70);
            if (message.includes('success')) setLoadingProgress(100);
          }
        );
        
        setModel(diffusion);
        setLoading(false);
      } catch (err) {
        console.error('Failed to load models:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    }
    
    loadModels();
  }, []);

  // Generate images
  const generate = useCallback(async () => {
    if (!model || generating) return;
    
    setGenerating(true);
    setGenerationStep(0);
    setError(null);
    
    const newImages: Array<{ imageData: ImageData; seed: number; classLabel: number; }> = [];
    
    try {
      const startTime = performance.now();
      
      for (let i = 0; i < numImages; i++) {
        const currentSeed = seed + i;
        
        const image = await model.generate(
          classLabel,
          numSteps,
          cfgScale,
          currentSeed,
          (step, total) => {
            setGenerationStep(i * numSteps + step);
          }
        );
        
        newImages.push({
          imageData: image,
          seed: currentSeed,
          classLabel: classLabel,
        });
      }
      
      const endTime = performance.now();
      console.log(`Generated ${numImages} images in ${(endTime - startTime).toFixed(0)}ms`);
      
      setGeneratedImages(newImages);
    } catch (err) {
      console.error('Generation failed:', err);
      setError(err instanceof Error ? err.message : 'Generation failed');
    } finally {
      setGenerating(false);
    }
  }, [model, generating, seed, numSteps, classLabel, cfgScale, numImages]);

  // Randomize seed
  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647));
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-600">
            🎨 Latent Diffusion
          </h1>
          <p className="text-gray-400 text-lg">
            AI image generation running entirely in your browser with ONNX Runtime
          </p>
        </header>

        {error && (
          <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 mb-8">
            <p className="text-red-300">⚠️ Error</p>
            <p className="text-sm text-red-200 mt-1">{error}</p>
            <p className="text-xs text-red-300 mt-2">
              Make sure the model files are available in the /public/models/ directory.
            </p>
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-8">
          {/* Controls Panel */}
          <div className="bg-gray-800/50 backdrop-blur rounded-2xl p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <span>⚙️</span> Generation Settings
            </h2>
            
            {/* Class Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Image Class
              </label>
              <div className="grid grid-cols-3 gap-2">
                {CLASS_OPTIONS.map((option) => (
                  <button
                    key={option.id}
                    onClick={() => setClassLabel(option.id)}
                    disabled={loading}
                    className={`p-3 rounded-xl transition-all ${
                      classLabel === option.id
                        ? 'bg-purple-600 ring-2 ring-purple-400'
                        : 'bg-gray-700 hover:bg-gray-600'
                    } disabled:opacity-50`}
                  >
                    <span className="text-2xl">{option.emoji}</span>
                    <p className="text-xs mt-1">{option.name.split(' ')[1]}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Number of Images */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Number of Images: {numImages}
              </label>
              <input
                type="range"
                min="1"
                max="12"
                value={numImages}
                onChange={(e) => setNumImages(parseInt(e.target.value))}
                disabled={loading}
                className="w-full accent-purple-500"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>1 image</span>
                <span>12 images</span>
              </div>
            </div>

            {/* Seed */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Starting Seed
              </label>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                  disabled={loading}
                  className="flex-1 bg-gray-700 rounded-lg px-4 py-2 text-white disabled:opacity-50"
                />
                <button
                  onClick={randomizeSeed}
                  disabled={loading}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50"
                  title="Randomize seed"
                >
                  🎲
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">Seeds: {seed} to {seed + numImages - 1}</p>
            </div>

            {/* Steps */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Diffusion Steps: {numSteps}
              </label>
              <input
                type="range"
                min="10"
                max="100"
                value={numSteps}
                onChange={(e) => setNumSteps(parseInt(e.target.value))}
                disabled={loading}
                className="w-full accent-purple-500"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Fast (10)</span>
                <span>Quality (100)</span>
              </div>
            </div>

            {/* CFG Scale */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                CFG Scale: {cfgScale.toFixed(1)}
              </label>
              <input
                type="range"
                min="1"
                max="15"
                step="0.5"
                value={cfgScale}
                onChange={(e) => setCfgScale(parseFloat(e.target.value))}
                disabled={loading}
                className="w-full accent-purple-500"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Creative (1)</span>
                <span>Strict (15)</span>
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={generate}
              disabled={loading || generating}
              className={`w-full py-4 rounded-xl font-semibold text-lg transition-all ${
                loading || generating
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 shadow-lg hover:shadow-purple-500/25'
              }`}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="animate-spin">⏳</span>
                  {loadingStatus}
                </span>
              ) : generating ? (
                <span>✨ Generating...</span>
              ) : (
                <span>✨ Generate Images</span>
              )}
            </button>

            {/* Generation Progress */}
            {generating && (
              <div className="mt-4">
                <div className="flex justify-between text-sm text-gray-300 mb-2">
                  <span>🎨 Image {Math.floor(generationStep / numSteps) + 1}/{numImages}</span>
                  <span>{Math.round((generationStep / (numSteps * numImages)) * 100)}%</span>
                </div>
                <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500 transition-all duration-200 animate-pulse"
                    style={{ width: `${(generationStep / (numSteps * numImages)) * 100}%` }}
                  />
                </div>
                <p className="text-xs text-gray-400 mt-2 text-center">
                  {generationStep === 0 ? 'Starting...' : 'Generating images...'}
                </p>
              </div>
            )}

            {/* Loading Progress */}
            {loading && (
              <div className="mt-4">
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
                    style={{ width: `${loadingProgress}%` }}
                  />
                </div>
                <p className="text-xs text-gray-400 mt-2 text-center">
                  {loadingProgress}% complete
                </p>
              </div>
            )}
          </div>

          {/* Output Panel */}
          <div className="bg-gray-800/50 backdrop-blur rounded-2xl p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <span>🖼️</span> Generated Images ({generatedImages.length})
            </h2>
            
            {generatedImages.length > 0 ? (
              <div className="grid grid-cols-3 gap-3 max-h-[600px] overflow-y-auto">
                {generatedImages.map((img, idx) => (
                  <div key={idx} className="relative group">
                    <canvas
                      ref={(el) => {
                        if (el) {
                          const ctx = el.getContext('2d');
                          if (ctx) {
                            ctx.putImageData(img.imageData, 0, 0);
                          }
                        }
                      }}
                      width={128}
                      height={128}
                      className="w-full h-auto rounded-lg border border-gray-600 image-rendering-pixelated"
                      style={{ imageRendering: 'pixelated' }}
                    />
                    <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex flex-col items-center justify-center gap-1 text-xs">
                      <p>{CLASS_OPTIONS[img.classLabel].emoji}</p>
                      <p className="text-gray-300">Seed: {img.seed}</p>
                      <button
                        onClick={() => {
                          const canvas = document.createElement('canvas');
                          canvas.width = 512;
                          canvas.height = 512;
                          const ctx = canvas.getContext('2d');
                          if (ctx) {
                            const tempCanvas = document.createElement('canvas');
                            tempCanvas.width = 128;
                            tempCanvas.height = 128;
                            const tempCtx = tempCanvas.getContext('2d');
                            if (tempCtx) {
                              tempCtx.putImageData(img.imageData, 0, 0);
                              ctx.imageSmoothingEnabled = false;
                              ctx.drawImage(tempCanvas, 0, 0, 512, 512);
                              const link = document.createElement('a');
                              link.download = `diffusion-${CLASS_OPTIONS[img.classLabel].name.split(' ')[1]}-${img.seed}.png`;
                              link.href = canvas.toDataURL('image/png');
                              link.click();
                            }
                          }
                        }}
                        className="mt-1 px-2 py-1 bg-purple-600 hover:bg-purple-500 rounded text-xs"
                      >
                        💾 Save
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="aspect-square bg-gray-900 rounded-xl flex items-center justify-center border border-gray-600">
                <div className="text-center text-gray-500">
                  <p className="text-6xl mb-4">🎨</p>
                  <p>Generated images will appear here</p>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>
            Running locally in your browser using ONNX Runtime WebAssembly.
          </p>
          <p className="mt-1">
            Models are downloaded once and cached by your browser.
          </p>
        </footer>
      </div>
    </main>
  );
}

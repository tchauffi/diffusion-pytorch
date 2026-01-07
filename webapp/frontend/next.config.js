/** @type {import('next').NextConfig} */
const path = require('path');

const nextConfig = {
  output: 'export',
  // Change this to your repo name for GitHub Pages
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
  images: {
    unoptimized: true,
  },
  // Enable WASM support
  webpack: (config, { isServer }) => {
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };
    
    // Fix for WASM in client-side
    if (!isServer) {
      // Force webpack to use browser field in package.json
      config.resolve.mainFields = ['browser', 'module', 'main'];
      
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
        stream: false,
        module: false,
      };
      
      // Exclude onnxruntime-node completely
      config.resolve.alias = {
        ...config.resolve.alias,
        'onnxruntime-node': false,
      };
    }
    
    // Disable critical warnings for dynamic requires
    config.module = {
      ...config.module,
      exprContextCritical: false,
      unknownContextCritical: false,
    };
    
    config.ignoreWarnings = [
      { module: /node_modules\/onnxruntime-web/ },
    ];
    
    return config;
  },
  // Headers for SharedArrayBuffer (needed for multi-threading)
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
          { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
        ],
      },
    ];
  },
};

module.exports = nextConfig;

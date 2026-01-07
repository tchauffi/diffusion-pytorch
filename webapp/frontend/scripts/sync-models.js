const fs = require('fs');
const path = require('path');
const https = require('https');

const MODELS = ['unet.onnx', 'vae-decoder.onnx'];
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const localModelDir = path.resolve(repoRoot, 'data', 'models_onnx');
const publicModelDir = path.resolve(__dirname, '..', 'public', 'models');
const baseUrl = process.env.HF_MODELS_BASE_URL || process.env.NEXT_PUBLIC_MODELS_BASE_URL;
const hfRepoId = process.env.HF_REPO_ID;
const hfBranch = process.env.HF_BRANCH || 'main';
const hfToken = process.env.HF_TOKEN || process.env.HF_ACCESS_TOKEN;

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function copyLocalModel(filename) {
  const sourcePath = path.join(localModelDir, filename);
  if (!fs.existsSync(sourcePath)) {
    return false;
  }
  const targetPath = path.join(publicModelDir, filename);
  fs.copyFileSync(sourcePath, targetPath);
  console.log(`Copied ${filename} from ${sourcePath}`);
  return true;
}

function downloadModel(filename, url) {
  return new Promise((resolve, reject) => {
    const targetPath = path.join(publicModelDir, filename);
    const fileStream = fs.createWriteStream(targetPath);
    const requestOptions = new URL(url);
    if (hfToken) {
      requestOptions.headers = {
        Authorization: `Bearer ${hfToken}`,
      };
    }
    const request = https.get(requestOptions, (response) => {
      if (response.statusCode && response.statusCode >= 400) {
        reject(new Error(`Failed to download ${filename}: ${response.statusCode}`));
        return;
      }
      response.pipe(fileStream);
    });
    request.on('error', reject);
    fileStream.on('finish', () => {
      fileStream.close(() => {
        console.log(`Downloaded ${filename} from ${url}`);
        resolve();
      });
    });
    fileStream.on('error', (err) => {
      fs.unlink(targetPath, () => reject(err));
    });
  });
}

async function ensureModel(filename) {
  const targetPath = path.join(publicModelDir, filename);
  if (fs.existsSync(targetPath)) {
    return;
  }

  if (copyLocalModel(filename)) {
    return;
  }

  let downloadUrl = baseUrl && `${baseUrl.replace(/\/$/, '')}/${filename}`;
  if (!downloadUrl && hfRepoId) {
    downloadUrl = `https://huggingface.co/${hfRepoId}/resolve/${hfBranch}/models/${filename}`;
  }

  if (downloadUrl) {
    await downloadModel(filename, downloadUrl);
    return;
  }

  throw new Error(
    `Missing ${filename}. Provide a local copy in ${localModelDir} or set HF_REPO_ID/HF_MODELS_BASE_URL.`
  );
}

(async () => {
  try {
    ensureDir(publicModelDir);
    for (const model of MODELS) {
      // eslint-disable-next-line no-await-in-loop
      await ensureModel(model);
    }
    console.log('Model sync complete.');
  } catch (err) {
    console.error(err.message || err);
    process.exit(1);
  }
})();

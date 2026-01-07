import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Latent Diffusion - AI Image Generator',
  description: 'Generate AI images using a Latent Diffusion Model running entirely in your browser via WebAssembly',
  keywords: ['AI', 'diffusion', 'image generation', 'WebAssembly', 'Rust'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}

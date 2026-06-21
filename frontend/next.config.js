/** @type {import('next').NextConfig} */
// Backend origin the /api/* proxy forwards to. Defaults to the local FastAPI
// server for dev; set BACKEND_URL in Vercel to your Cloudflare tunnel URL.
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';

const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${BACKEND_URL}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;

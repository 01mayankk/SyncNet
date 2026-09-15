/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          dark: '#0A0E17',
          card: '#0F172A',
          cyan: '#06B6D4',
          emerald: '#10B981',
          rose: '#EF4444',
          amber: '#F59E0B',
          purple: '#8B5CF6',
          blue: '#3B82F6',
        },
      },
    },
  },
  plugins: [],
};

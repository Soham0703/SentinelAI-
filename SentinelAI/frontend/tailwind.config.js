export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0B0F17',
          850: '#111827',
          800: '#1F2937',
          700: '#374151'
        }
      }
    },
  },
  plugins: [],
}

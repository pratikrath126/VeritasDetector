import { useEffect, useMemo, useState } from 'react';
import { analyzeImage, checkHealth } from './api';
import UploadZone from './components/UploadZone';
import ResultCard from './components/ResultCard';

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [backendStatus, setBackendStatus] = useState({ node_server: 'offline', ml_engine: 'offline' });

  useEffect(() => {
    let active = true;

    const runHealthCheck = async () => {
      const status = await checkHealth();
      if (active) setBackendStatus(status);
    };

    runHealthCheck();
    const timer = setInterval(runHealthCheck, 10000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const statusView = useMemo(() => {
    if (backendStatus.node_server !== 'online') {
      return { dot: 'bg-yellow-400', text: 'Backend Offline' };
    }
    if (backendStatus.ml_engine !== 'online') {
      return { dot: 'bg-red-500', text: 'ML Engine Offline' };
    }
    return { dot: 'bg-green-500', text: 'Systems Online' };
  }, [backendStatus]);

  const onFilePicked = (picked) => {
    if (!picked) return;
    setError('');
    setResult(null);
    if (preview) URL.revokeObjectURL(preview);
    setFile(picked);
    setPreview(URL.createObjectURL(picked));
  };

  const handleFileChange = (e) => {
    onFilePicked(e.target.files?.[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped && dropped.type.startsWith('image/')) onFilePicked(dropped);
  };

  const handleAnalyze = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError('');

    try {
      const apiResult = await analyzeImage(file);
      setResult(apiResult);
    } catch (err) {
      const message = err?.response?.data?.message || err?.response?.data?.error || err.message || 'Analysis failed.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError('');
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    const input = document.getElementById('fileInput');
    if (input) input.value = '';
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white px-4 py-6">
      <header className="max-w-5xl mx-auto flex items-center justify-between mb-8">
        <h1 className="text-3xl md:text-4xl font-bold tracking-widest">VERITAS</h1>
        <div className="flex items-center gap-2 text-sm">
          <span className={`w-3 h-3 rounded-full ${statusView.dot}`} />
          <span className="text-gray-200">{statusView.text}</span>
        </div>
      </header>

      <main className="max-w-3xl mx-auto">
        {!result && (
          <>
            <section className="text-center mb-6">
              <h2 className="text-2xl md:text-3xl font-bold">DEEPFAKE FACE DETECTION</h2>
              <p className="text-gray-400 mt-2">Upload any face photo to analyze authenticity</p>
            </section>

            <UploadZone
              file={file}
              preview={preview}
              isDragging={isDragging}
              loading={loading}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onBrowse={() => document.getElementById('fileInput')?.click()}
              onFileChange={handleFileChange}
            />

            {file && (
              <button
                type="button"
                disabled={loading}
                onClick={handleAnalyze}
                className="w-full mt-4 rounded-lg bg-green-600 hover:bg-green-500 disabled:bg-green-900 disabled:cursor-not-allowed transition-colors py-4 text-lg font-bold"
              >
                {loading ? 'SCANNING...' : 'ANALYZE'}
              </button>
            )}
          </>
        )}

        {result && <ResultCard result={result} preview={preview} onReset={handleReset} />}

        {error && (
          <div className="mt-5 rounded-xl border border-red-500 bg-red-950/40 p-4">
            <div className="font-bold text-red-300">Error</div>
            <div className="text-red-200 mt-1">{error}</div>
            {error.toLowerCase().includes('ml service') && (
              <div className="text-yellow-300 text-sm mt-2">
                Python ML Engine is offline. Start it with: <code>cd ml_engine && python api.py</code>
              </div>
            )}
            <button
              type="button"
              onClick={() => setError('')}
              className="mt-3 px-4 py-2 bg-red-700 hover:bg-red-600 rounded text-white"
            >
              Try Again
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

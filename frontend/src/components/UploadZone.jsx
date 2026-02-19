import LoadingScanner from './LoadingScanner';

export default function UploadZone({
  file,
  preview,
  isDragging,
  loading,
  onDragOver,
  onDragLeave,
  onDrop,
  onBrowse,
  onFileChange,
}) {
  return (
    <div
      className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
        loading ? 'pulse-border' : ''
      } ${isDragging ? 'border-blue-400 bg-blue-950/20' : 'border-gray-600 hover:border-blue-500'}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onBrowse}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onBrowse()}
    >
      {!file ? (
        <>
          <div className="text-4xl mb-3">📤</div>
          <div className="text-xl text-white font-bold">Drag & drop face image here</div>
          <div className="text-gray-400 mt-2">or click to browse</div>
          <div className="text-gray-500 text-sm mt-1">Supports JPG, PNG, WebP</div>
        </>
      ) : (
        <>
          <img
            src={preview}
            alt="Uploaded preview"
            className="mx-auto max-h-[300px] rounded-lg border border-gray-600"
          />
          <div className="mt-3 text-gray-300 break-all">{file.name}</div>
        </>
      )}

      {loading && <div className="mt-4"><LoadingScanner /></div>}

      <input
        type="file"
        accept="image/*"
        className="hidden"
        id="fileInput"
        onChange={onFileChange}
      />
    </div>
  );
}

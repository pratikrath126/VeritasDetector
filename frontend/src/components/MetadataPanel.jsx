function getStatusInfo(status) {
  if (status === 'Likely Real') return { icon: '✅', color: 'text-green-400' };
  if (status === 'Suspicious') return { icon: '⚠️', color: 'text-yellow-400' };
  if (status === 'Likely Fake') return { icon: '❌', color: 'text-red-400' };
  return { icon: '❓', color: 'text-gray-400' };
}

export default function MetadataPanel({ metadata = {} }) {
  const status = metadata.status || 'Inconclusive';
  const reason = metadata.reason || 'No metadata analysis available.';
  const details = metadata.details || {};
  const { icon, color } = getStatusInfo(status);

  return (
    <div className="rounded-xl border border-gray-700 bg-[#111] p-5 h-full">
      <div className="text-sm text-gray-300 font-bold mb-4">🔍 METADATA ANALYSIS</div>
      <div className={`text-4xl mb-2 ${color}`}>{icon}</div>
      <div className={`font-bold text-lg ${color}`}>{status}</div>
      <p className="text-sm text-gray-400 mt-2">{reason}</p>

      <hr className="my-4 border-gray-700" />

      <div className="grid grid-cols-2 gap-3 text-xs">
        <div>
          <div className="text-gray-500">Camera</div>
          <div className="text-gray-200 break-words">{details.camera_make || 'Not found'}</div>
        </div>
        <div>
          <div className="text-gray-500">Model</div>
          <div className="text-gray-200 break-words">{details.camera_model || 'Not found'}</div>
        </div>
        <div>
          <div className="text-gray-500">Software</div>
          <div className="text-gray-200 break-words">{details.software || 'Not found'}</div>
        </div>
        <div>
          <div className="text-gray-500">GPS</div>
          <div className="text-gray-200">{details.has_gps ? 'Yes' : 'No'}</div>
        </div>
      </div>

      {(status === 'Suspicious' || status === 'Likely Fake') && (
        <div className={`mt-4 rounded-lg p-3 text-xs ${status === 'Suspicious' ? 'bg-yellow-950 text-yellow-200' : 'bg-red-950 text-red-200'}`}>
          {status === 'Suspicious'
            ? 'AI images typically have no camera metadata'
            : 'This image shows signs of AI generation'}
        </div>
      )}
    </div>
  );
}

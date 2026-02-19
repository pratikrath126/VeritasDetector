import ConfidenceBar from './ConfidenceBar';
import MetadataPanel from './MetadataPanel';

function getSummary(result) {
  const mlFake = result.label === 'Fake';
  const metaStatus = result?.metadata?.status;

  if (mlFake && metaStatus === 'Suspicious') return 'HIGH CONFIDENCE AI GENERATED';
  if (!mlFake && metaStatus === 'Likely Real') return 'HIGH CONFIDENCE REAL PERSON';
  return 'ANALYSIS COMPLETE — REVIEW RESULTS ABOVE';
}

export default function ResultCard({ result, preview, onReset }) {
  const isFake = result.label === 'Fake';
  const verdictClass = isFake ? 'bg-red-600' : 'bg-green-600';
  const verdictText = isFake ? '⚠ AI GENERATED FACE DETECTED' : '✓ REAL PERSON DETECTED';
  const accentClass = isFake ? 'text-red-400' : 'text-green-400';
  const borderClass = isFake ? 'border-red-500' : 'border-green-500';

  return (
    <div className="fade-in space-y-5">
      <div className={`${verdictClass} text-white rounded-xl p-6 text-2xl font-bold tracking-wide`}>
        {verdictText}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="rounded-xl border border-gray-700 bg-[#111] p-5">
          <img
            src={preview}
            alt="Analyzed"
            className={`w-full rounded-lg border-2 ${borderClass}`}
          />
          <div className="mt-4 text-sm text-gray-300">CONFIDENCE SCORE</div>
          <div className={`text-5xl font-bold mt-1 ${accentClass}`}>{result.confidence}%</div>
          <div className="mt-3">
            <ConfidenceBar confidence={result.confidence} isFake={isFake} />
          </div>
          <div className="mt-2 text-sm text-gray-400">
            Real: {result?.scores?.real ?? 0}% | Fake: {result?.scores?.fake ?? 0}%
          </div>
        </div>

        <MetadataPanel metadata={result.metadata} />
      </div>

      <div className="rounded-xl border border-gray-700 bg-[#111] p-5">
        <div className="text-sm text-gray-300">OVERALL VERDICT SUMMARY</div>
        <div className="text-xl text-white font-bold mt-2">{getSummary(result)}</div>
      </div>

      <button
        type="button"
        onClick={onReset}
        className="w-full rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors py-3 font-bold text-white"
      >
        ANALYZE ANOTHER IMAGE
      </button>
    </div>
  );
}

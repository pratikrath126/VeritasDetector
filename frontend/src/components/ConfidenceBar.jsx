export default function ConfidenceBar({ confidence, isFake }) {
  const fillClass = isFake ? 'bg-red-500' : 'bg-green-500';

  return (
    <div className="w-full bg-gray-800 rounded-full h-4 overflow-hidden border border-gray-700">
      <div
        className={`${fillClass} h-4 transition-all duration-700 ease-out`}
        style={{ width: `${Math.max(0, Math.min(confidence, 100))}%` }}
      />
    </div>
  );
}

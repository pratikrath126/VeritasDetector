import { useEffect, useState } from 'react';

export default function LoadingScanner() {
  const [dots, setDots] = useState('');

  useEffect(() => {
    const timer = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? '' : `${prev}.`));
    }, 350);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="text-center text-blue-300 font-bold tracking-widest">
      SCANNING{dots}
    </div>
  );
}

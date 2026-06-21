'use client';

import { useEffect, useState } from 'react';

// Reports whether the viewport is phone-sized. Starts false (matches SSR /
// desktop), then updates on mount and on resize. Breakpoint matches Tailwind md.
export function useIsMobile(breakpoint = 768): boolean {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${breakpoint - 1}px)`);
    const update = () => setIsMobile(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, [breakpoint]);

  return isMobile;
}

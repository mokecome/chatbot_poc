import { useEffect, useState, ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface FixedPositionPortalProps {
  children: ReactNode;
}

/**
 * Portal component that renders children directly to document.body
 * This ensures that position: fixed elements work correctly regardless of parent containers
 * that might create new stacking contexts (transform, filter, will-change, etc.)
 */
export function FixedPositionPortal({ children }: FixedPositionPortalProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  // Only render on client side to avoid SSR issues
  if (!mounted) return null;

  // Render children directly to document.body
  return createPortal(children, document.body);
}
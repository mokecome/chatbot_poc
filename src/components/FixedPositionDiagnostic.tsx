/**
 * CSS Properties that create new containing blocks and affect position: fixed
 * 
 * The following CSS properties can cause position: fixed elements to be positioned
 * relative to their containing element instead of the viewport:
 * 
 * 1. transform: any value other than 'none'
 * 2. will-change: transform
 * 3. perspective: any value other than 'none'  
 * 4. filter: any value other than 'none'
 * 5. contain: layout, style, paint, or strict
 * 6. backdrop-filter: any value other than 'none'
 * 
 * Diagnostic function to check if an element has properties affecting position: fixed
 */

export function diagnoseFixedPositioning(element: HTMLElement): {
  hasIssues: boolean;
  problematicProperties: string[];
  suggestions: string[];
} {
  const computedStyle = window.getComputedStyle(element);
  const problematicProperties: string[] = [];
  const suggestions: string[] = [];

  // Check transform
  if (computedStyle.transform !== 'none') {
    problematicProperties.push(`transform: ${computedStyle.transform}`);
    suggestions.push('Remove transform property or move fixed element outside this container');
  }

  // Check will-change
  const willChange = computedStyle.getPropertyValue('will-change');
  if (willChange.includes('transform')) {
    problematicProperties.push(`will-change: ${willChange}`);
    suggestions.push('Remove will-change: transform or move fixed element outside this container');
  }

  // Check perspective
  if (computedStyle.perspective !== 'none') {
    problematicProperties.push(`perspective: ${computedStyle.perspective}`);
    suggestions.push('Remove perspective property or move fixed element outside this container');
  }

  // Check filter
  if (computedStyle.filter !== 'none') {
    problematicProperties.push(`filter: ${computedStyle.filter}`);
    suggestions.push('Remove filter property or move fixed element outside this container');
  }

  // Check backdrop-filter
  const backdropFilter = computedStyle.getPropertyValue('backdrop-filter');
  if (backdropFilter && backdropFilter !== 'none') {
    problematicProperties.push(`backdrop-filter: ${backdropFilter}`);
    suggestions.push('Remove backdrop-filter property or move fixed element outside this container');
  }

  // Check contain
  const contain = computedStyle.getPropertyValue('contain');
  if (contain && contain !== 'none' && (
    contain.includes('layout') || 
    contain.includes('style') || 
    contain.includes('paint') || 
    contain.includes('strict')
  )) {
    problematicProperties.push(`contain: ${contain}`);
    suggestions.push('Remove contain property or move fixed element outside this container');
  }

  return {
    hasIssues: problematicProperties.length > 0,
    problematicProperties,
    suggestions
  };
}

/**
 * Hook to automatically diagnose fixed positioning issues for an element
 */
export function useDiagnoseFixedPositioning(elementRef: React.RefObject<HTMLElement>) {
  const [diagnosis, setDiagnosis] = React.useState<ReturnType<typeof diagnoseFixedPositioning> | null>(null);

  React.useEffect(() => {
    if (elementRef.current) {
      let current = elementRef.current.parentElement;
      const allIssues: string[] = [];
      const allSuggestions: string[] = [];

      // Check all ancestor elements
      while (current && current !== document.body) {
        const result = diagnoseFixedPositioning(current);
        if (result.hasIssues) {
          allIssues.push(...result.problematicProperties.map(prop => 
            `${current.tagName.toLowerCase()}${current.className ? '.' + current.className.replace(/\s+/g, '.') : ''}: ${prop}`
          ));
          allSuggestions.push(...result.suggestions);
        }
        current = current.parentElement;
      }

      setDiagnosis({
        hasIssues: allIssues.length > 0,
        problematicProperties: allIssues,
        suggestions: Array.from(new Set(allSuggestions)) // Remove duplicates
      });
    }
  }, [elementRef]);

  return diagnosis;
}

// Add React import for the hook
import React from 'react';
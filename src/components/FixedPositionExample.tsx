import { Button } from "./ui/button";
import { FixedPositionPortal } from "./FixedPositionPortal";

/**
 * Example component demonstrating proper fixed positioning techniques
 * 
 * Common issues and solutions:
 * 1. Parent container with transform/filter/will-change creates new containing block
 * 2. Solution: Use React Portal to render fixed elements directly to document.body
 */

export function FixedPositionExample() {
  return (
    <div className="p-8">
      <h2 className="text-xl font-semibold mb-6">Fixed Position Examples</h2>
      
      {/* Problem: Fixed button inside container with transform */}
      <div className="mb-8">
        <h3 className="text-lg font-medium mb-2 text-red-600">❌ Problem: Fixed button affected by parent transform</h3>
        <div className="relative p-6 bg-gray-100 rounded-lg transform scale-105">
          <p>This container has transform: scale(1.05)</p>
          <Button className="fixed bottom-4 right-4 bg-red-500 hover:bg-red-600">
            Broken Fixed Button
          </Button>
        </div>
      </div>

      {/* Solution: Using Portal */}
      <div className="mb-8">
        <h3 className="text-lg font-medium mb-2 text-green-600">✅ Solution: Fixed button using Portal</h3>
        <div className="relative p-6 bg-gray-100 rounded-lg transform scale-105">
          <p>This container has transform: scale(1.05)</p>
          <FixedPositionPortal>
            <Button className="fixed bottom-20 right-4 bg-green-500 hover:bg-green-600 z-[9999]">
              Fixed Button (Portal)
            </Button>
          </FixedPositionPortal>
        </div>
      </div>

      {/* Other problematic properties */}
      <div className="space-y-4">
        <h3 className="text-lg font-medium">CSS Properties that break position: fixed:</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
            <strong>transform:</strong> any value other than 'none'<br/>
            <code className="text-xs bg-yellow-100 px-1 rounded">transform: translateX(10px)</code>
          </div>
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
            <strong>filter:</strong> any value other than 'none'<br/>
            <code className="text-xs bg-yellow-100 px-1 rounded">filter: blur(2px)</code>
          </div>
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
            <strong>will-change:</strong> transform<br/>
            <code className="text-xs bg-yellow-100 px-1 rounded">will-change: transform</code>
          </div>
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
            <strong>perspective:</strong> any value other than 'none'<br/>
            <code className="text-xs bg-yellow-100 px-1 rounded">perspective: 1000px</code>
          </div>
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
            <strong>backdrop-filter:</strong> any value other than 'none'<br/>
            <code className="text-xs bg-yellow-100 px-1 rounded">backdrop-filter: blur(10px)</code>
          </div>
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
            <strong>contain:</strong> layout, style, paint, or strict<br/>
            <code className="text-xs bg-yellow-100 px-1 rounded">contain: layout style</code>
          </div>
        </div>
      </div>

      {/* Solutions */}
      <div className="mt-8">
        <h3 className="text-lg font-medium mb-4">Solutions:</h3>
        <div className="space-y-3 text-sm">
          <div className="p-3 bg-green-50 border border-green-200 rounded">
            <strong>1. Use React Portal:</strong> Render fixed elements directly to document.body
          </div>
          <div className="p-3 bg-green-50 border border-green-200 rounded">
            <strong>2. Move element outside container:</strong> Place fixed elements outside problematic containers
          </div>
          <div className="p-3 bg-green-50 border border-green-200 rounded">
            <strong>3. Remove problematic CSS:</strong> If possible, remove transform/filter properties from parent containers
          </div>
          <div className="p-3 bg-green-50 border border-green-200 rounded">
            <strong>4. Use absolute positioning:</strong> If fixed behavior isn't strictly required, use absolute with proper container
          </div>
        </div>
      </div>
    </div>
  );
}
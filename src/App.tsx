import { useState } from "react";
import { FloatingChatbot } from "./components/FloatingChatbot";
import { ScrollBarTest } from "./components/ScrollBarTest";
import { Button } from "./components/ui/button";

export default function App() {
  const [showTest, setShowTest] = useState(false);

  return (
    <>
      <div className="fixed top-4 left-4 z-[10000]">
        <Button
          onClick={() => setShowTest(!showTest)}
          variant={showTest ? "default" : "outline"}
          size="sm"
        >
          {showTest ? "Back to Chat" : "Open Scroll Demo"}
        </Button>
      </div>

      {showTest ? (
        <div className="fixed inset-0 z-[10000] flex justify-end items-start pointer-events-none">
          <div className="pointer-events-auto mt-20 mr-6 max-w-[min(90vw,1000px)] max-h-[calc(100vh-6rem)] overflow-y-auto rounded-lg shadow-xl bg-white">
            <ScrollBarTest />
          </div>
        </div>
      ) : (
        <FloatingChatbot />
      )}
    </>
  );
}

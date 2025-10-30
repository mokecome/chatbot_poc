export function HotelBackground() {
  return (
    <div className="relative w-full h-screen overflow-hidden bg-[#151120]">
      <div className="absolute inset-0 bg-gradient-to-br from-[#151120] via-[#221B33] to-[#382747]" />
      <div
        className="absolute inset-0 opacity-20 mix-blend-screen"
        style={{
          backgroundImage:
            "radial-gradient(circle at 15% 20%, rgba(255, 255, 255, 0.2) 0, transparent 40%), radial-gradient(circle at 85% 30%, rgba(255, 228, 200, 0.25) 0, transparent 45%), radial-gradient(circle at 30% 75%, rgba(255, 255, 255, 0.18) 0, transparent 50%)",
        }}
      />
      <div
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage:
            "repeating-linear-gradient(45deg, transparent 0 12px, rgba(255, 255, 255, 0.05) 12px 24px)",
        }}
      />

      <div className="relative z-10 flex flex-col items-center justify-center h-full text-center text-white space-y-6 px-6">
        <div className="space-y-3">
          <p className="text-sm uppercase tracking-[0.5em] text-white/50">
            Est. 1998
          </p>
          <h1 className="text-4xl md:text-5xl font-semibold tracking-wide">
            水漾月明度假文旅
          </h1>
          <p className="max-w-xl text-base md:text-lg text-white/75">
            湖畔景緻入住體驗 · 在地風味餐飲 · 專屬禮賓服務
          </p>
        </div>
        <div className="flex flex-wrap justify-center gap-4 text-sm text-white/60">
          <span>豪華客房與套房</span>
          <span>全天候禮賓管家</span>
          <span>空中酒吧 · 星辰泳池</span>
          <span>商旅會議設備</span>
        </div>
      </div>

      <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-[#151120] to-transparent" />
    </div>
  );
}

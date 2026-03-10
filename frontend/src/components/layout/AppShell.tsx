// src/components/layout/AppShell.tsx
// 4-column layout: VerticalToolbar | ContextOptionsPanel | Viewport | PropertiesPanel
import React from "react";
import HeaderBar from "./HeaderBar";
import StatusBar from "./StatusBar";
import VerticalToolbar from "./VerticalToolbar";
import ContextOptionsPanel from "./ContextOptionsPanel";
import PropertiesPanel from "./PropertiesPanel";
import { useUiStore, type WorkflowStep } from "@/store/uiStore";

// Step panel imports
import UploadPanel from "@/components/sidebar/UploadPanel";
import ScoutPanel from "@/components/sidebar/ScoutPanel";
import SelectPanel from "@/components/sidebar/SelectPanel";
import AISegPanel from "@/components/sidebar/AISegPanel";
import RefinePanel from "@/components/sidebar/RefinePanel";
import MeshPanel from "@/components/sidebar/MeshPanel";
import ExportPanel from "@/components/sidebar/ExportPanel";

// Viewport
import ViewportScene from "@/components/viewport/ViewportScene";

// Step config
interface StepPanelConfig {
  readonly title: string;
  readonly icon: string;
  readonly component: React.FC;
}

const STEP_PANELS: Readonly<Record<WorkflowStep, StepPanelConfig>> = {
  upload:  { title: "Upload Study",      icon: "", component: UploadPanel },
  scout:   { title: "Scout Anatomy",     icon: "", component: ScoutPanel },
  select:  { title: "Select & Crop ROI", icon: "", component: SelectPanel },
  segment: { title: "AI Segmentation",   icon: "", component: AISegPanel },
  refine:  { title: "Interactive Refine",icon: "",  component: RefinePanel },
  mesh:    { title: "Generate Mesh",     icon: "", component: MeshPanel },
  export:  { title: "Export",            icon: "", component: ExportPanel },
} as const;

const AppShell: React.FC = () => {
  const activeStep = useUiStore((s) => s.activeStep);
  const panelConfig = STEP_PANELS[activeStep];
  const StepContent = panelConfig.component;

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-background">
      <HeaderBar />

      {/* Main 4-column body */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Col 1: Vertical toolbar (w-14, fixed) */}
        <VerticalToolbar />

        {/* Col 2: Context options panel (320px collapsible) */}
        <ContextOptionsPanel
          stepTitle={panelConfig.title}
          stepIcon={<span aria-hidden>{panelConfig.icon}</span>}
        >
          <StepContent />
        </ContextOptionsPanel>

        {/* Col 3: Main 3D viewport (flex-1) */}
        <main className="flex-1 min-w-0 relative overflow-hidden">
          <ViewportScene />
        </main>

        {/* Col 4: Properties panel (280px collapsible) */}
        <PropertiesPanel />
      </div>

      <StatusBar />
    </div>
  );
};

export default AppShell;

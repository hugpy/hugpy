export default function useColumnLayout(storageKey: any, defaultKeys: any): {
    order: any[];
    widths: {};
    moveColumn: (key: any, beforeKey: any) => void;
    setWidth: (key: any, px: any) => void;
    reset: () => void;
};

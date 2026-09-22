export function onDemoToast(fn: any): () => void;
export function demoFetch(input: any, init: any): Promise<Response>;
export function installDemoMode(): void;
export function uninstallDemoMode(): void;
export const DEMO_MSG: "This is a demo \u2014 install hugpy (pip install hugpy) and run the console to do this for real.";

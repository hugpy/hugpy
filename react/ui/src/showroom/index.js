// Public surface of the showroom demo-mode module.
export { default } from './ShowroomConsole.jsx'
export { default as ShowroomConsole } from './ShowroomConsole.jsx'
export { installDemoMode, uninstallDemoMode, onDemoToast, demoFetch, DEMO_MSG } from './demoFetch.js'
export { installDemoModeIfDemoHost } from './earlyInstall.js'

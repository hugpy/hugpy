import { default as React } from 'react';
import { HugpyRuntimeConfig } from './config';
export interface HugpyProviderProps extends Partial<HugpyRuntimeConfig> {
    children?: React.ReactNode;
}
export declare function HugpyProvider({ children, ...cfg }: HugpyProviderProps): React.JSX.Element;
/** Read the active runtime config reactively from within the provider. */
export declare function useHugpyConfig(): HugpyRuntimeConfig;

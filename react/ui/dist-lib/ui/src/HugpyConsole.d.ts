import { default as React } from 'react';
import { HugpyProviderProps } from './runtime/HugpyProvider';
export type HugpyConsoleProps = Omit<HugpyProviderProps, 'children'>;
export declare function HugpyConsole(props: HugpyConsoleProps): React.JSX.Element;

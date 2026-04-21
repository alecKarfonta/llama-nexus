declare module 'react' {
  // Hooks
  export const useState: any;
  export const useRef: any;
  export const useEffect: any;
  export const useCallback: any;
  export const useMemo: any;
  export const useContext: any;
  export const useImperativeHandle: any;

  // Utilities
  export const createContext: any;
  export const forwardRef: any;
  export const memo: any;

  // Types
  export type ChangeEvent<T = any> = any;
  export type KeyboardEvent<T = any> = any;
  export type ReactNode = any;
  export type ErrorInfo = any;
  export type FC<P = {}> = (props: P) => any;
  export type FunctionComponent<P = {}> = (props: P) => any;

  // Class component
  export class Component<P = {}, S = {}> {
    props: P;
    state: S;
    setState(state: Partial<S>): void;
    render(): any;
  }
}

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}

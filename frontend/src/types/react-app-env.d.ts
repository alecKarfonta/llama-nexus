/// <reference types="react" />
/// <reference types="react-dom" />

declare module 'react' {
  interface JSX {
    IntrinsicElements: any;
  }
}

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}

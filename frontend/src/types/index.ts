/**
 * Main type exports for the application
 */

export * from './api';

// UI-specific types
export interface Theme {
  mode: 'light' | 'dark';
}

export interface AppError {
  id: string;
  message: string;
  type: 'error' | 'warning' | 'info';
  timestamp: Date;
  dismissible: boolean;
}

export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
  badge?: string | number;
  color?: string;
}

export interface NavigationSection {
  id: string;
  title: string;
  items: NavigationItem[];
}

// Form types
export interface FormField {
  name: string;
  label: string;
  type: 'text' | 'number' | 'select' | 'slider' | 'boolean';
  value: any;
  required?: boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: { label: string; value: any }[];
  description?: string;
  validation?: (value: any) => string | undefined;
}

// Chart data types
export interface ChartDataPoint {
  timestamp: Date;
  value: number;
  label?: string;
}

export interface ChartSeries {
  name: string;
  data: ChartDataPoint[];
  color?: string;
  unit?: string;
}

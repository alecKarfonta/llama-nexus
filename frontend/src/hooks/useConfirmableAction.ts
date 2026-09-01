import { useCallback, useMemo, useState } from 'react';
import type { ConfirmationAction } from '@/components/common';
import { useLocalPreference } from '@/hooks/useLocalPreferences';

interface UseConfirmableActionOptions {
  onError?: (error: unknown) => void;
}

export const useConfirmableAction = (options: UseConfirmableActionOptions = {}) => {
  const { onError } = options;
  const { value: confirmDestructiveActions } = useLocalPreference('confirmDestructiveActions');
  const [confirmationAction, setConfirmationAction] = useState<ConfirmationAction | null>(null);
  const [confirmationLoading, setConfirmationLoading] = useState(false);

  const closeConfirmation = useCallback(() => {
    if (confirmationLoading) return;
    setConfirmationAction(null);
  }, [confirmationLoading]);

  const executeAction = useCallback(async (action: ConfirmationAction) => {
    try {
      setConfirmationLoading(true);
      await action.onConfirm();
      setConfirmationAction(null);
    } catch (error) {
      onError?.(error);
    } finally {
      setConfirmationLoading(false);
    }
  }, [onError]);

  const confirmAction = useCallback(async () => {
    if (!confirmationAction) return;
    await executeAction(confirmationAction);
  }, [confirmationAction, executeAction]);

  const requestConfirmation = useCallback(async (action: ConfirmationAction) => {
    if (action.destructive && !confirmDestructiveActions) {
      await executeAction(action);
      return;
    }

    setConfirmationAction(action);
  }, [confirmDestructiveActions, executeAction]);

  return useMemo(() => ({
    confirmationAction,
    confirmationLoading,
    requestConfirmation,
    closeConfirmation,
    confirmAction,
  }), [
    confirmationAction,
    confirmationLoading,
    requestConfirmation,
    closeConfirmation,
    confirmAction,
  ]);
};

import { useState, useCallback } from "react";

/**
 * Modal state using discriminated union pattern.
 * Only one modal can be open at a time, ensuring consistent UI behavior.
 */
export type ModalState =
  | { type: "none" }
  | { type: "missionComplete"; outcome?: "victory" | "defeat" }
  | { type: "forfeit" }
  | { type: "pause" }
  | { type: "settings" }
  | { type: "overcharge" }
  | { type: "help" }
  | { type: "tutorial" }
  | { type: "navigationConfirm"; pendingPath?: string };

/**
 * Helper type to extract modal data for a given modal type.
 */
export type ModalDataFor<T extends ModalState["type"]> = Extract<
  ModalState,
  { type: T }
>;

/**
 * Hook for managing modal state with type-safe open/close functions.
 * Ensures only one modal is open at a time using a discriminated union.
 *
 * @example
 * ```tsx
 * const { modal, openModal, closeModal, isOpen } = useModalManager();
 *
 * // Open a modal
 * openModal({ type: "missionComplete", outcome: "victory" });
 *
 * // Check if a specific modal is open
 * if (isOpen("missionComplete")) {
 *   // modal.outcome is available here
 * }
 *
 * // Close current modal
 * closeModal();
 * ```
 */
export function useModalManager() {
  const [modal, setModal] = useState<ModalState>({ type: "none" });

  /**
   * Open a modal. Closes any currently open modal.
   */
  const openModal = useCallback((state: Exclude<ModalState, { type: "none" }>) => {
    setModal(state);
  }, []);

  /**
   * Close the current modal (reset to none).
   */
  const closeModal = useCallback(() => {
    setModal({ type: "none" });
  }, []);

  /**
   * Check if a specific modal type is open.
   * Returns true if the modal type matches.
   */
  const isOpen = useCallback(
    <T extends ModalState["type"]>(type: T): boolean => {
      return modal.type === type;
    },
    [modal.type]
  );

  /**
   * Get the current modal type (for quick checks without type narrowing).
   */
  const modalType = modal.type;

  /**
   * Check if any modal is currently open (useful for keyboard guards).
   */
  const isAnyModalOpen = modal.type !== "none";

  return {
    /** Current modal state */
    modal,
    /** Open a modal (closes any currently open modal) */
    openModal,
    /** Close the current modal */
    closeModal,
    /** Type guard for checking if a specific modal is open */
    isOpen,
    /** Current modal type for quick checks */
    modalType,
    /** Whether any modal is currently open */
    isAnyModalOpen,
  };
}

/**
 * Type for the return value of useModalManager hook.
 */
export type ModalManager = ReturnType<typeof useModalManager>;

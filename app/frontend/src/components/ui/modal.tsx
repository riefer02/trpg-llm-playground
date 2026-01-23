import { useEffect } from "react";

export interface ModalProps {
  isOpen: boolean;
  onClose?: () => void;
  children: React.ReactNode;
  /** If true, clicking backdrop won't close modal */
  disableBackdropClose?: boolean;
  /** Adds pulsing border animation for urgent prompts */
  urgent?: boolean;
}

/**
 * Modal overlay component for important prompts.
 * Renders children in a centered card with backdrop.
 */
export function Modal({
  isOpen,
  onClose,
  children,
  disableBackdropClose = false,
  urgent = false,
}: ModalProps) {
  // Lock body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = "";
      };
    }
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    if (!isOpen || disableBackdropClose) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && onClose) {
        onClose();
      }
    };

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose, disableBackdropClose]);

  if (!isOpen) return null;

  const handleBackdropClick = () => {
    if (!disableBackdropClose && onClose) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      role="dialog"
      aria-modal="true"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
        onClick={handleBackdropClick}
        aria-hidden="true"
      />

      {/* Modal content */}
      <div
        className={`relative z-10 max-w-lg w-full mx-4 animate-in zoom-in-95 fade-in duration-200 ${
          urgent ? "animate-pulse-border" : ""
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}

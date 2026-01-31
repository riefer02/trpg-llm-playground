import { KeyboardShortcutsModal } from "./ui/keyboard-shortcuts-modal";
import { useKeyboardShortcuts } from "../lib/hooks/useKeyboardShortcuts";

export function KeyboardShortcutsManager() {
  const { isOpen, close } = useKeyboardShortcuts();

  return <KeyboardShortcutsModal isOpen={isOpen} onClose={close} />;
}
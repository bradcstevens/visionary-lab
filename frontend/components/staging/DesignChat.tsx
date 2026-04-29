"use client"

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Sparkles, FileText, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { QuickReplyChips } from "./QuickReplyChips";
import { chatWithProject, ChatMessage, ChatResponse } from "@/services/stagingApi";

/** Phrases that indicate the user wants to proceed to brief generation. */
const PROCEED_PATTERNS = [
  'go ahead', 'proceed', 'generate brief', 'generate the brief', 'create brief',
  'create the brief', 'looks good', "let's go", "let's do it", "let's proceed",
  'move on', 'next step', "i'm happy", "i'm ready", "that's great", "thats great",
  'perfect', 'sounds good', 'do it', 'ready to go', 'good to go', "let's move on",
  "move to the brief", "move to brief", "design brief", "make the brief",
];

function isProceedIntent(message: string): boolean {
  const lower = message.toLowerCase().trim();
  return PROCEED_PATTERNS.some(phrase => lower.includes(phrase));
}

interface DesignChatProps {
  projectId: string;
  focusedImageId: string | null;
  focusedImageLabel: string | null;
  onClearFocus: () => void;
  onReadyForBrief: () => void;
  isGeneratingBrief?: boolean;
  canGenerateBrief?: boolean;
  initialMessage: string;
  conversationHistory: ChatMessage[];
  onHistoryUpdate: (history: ChatMessage[]) => void;
}

export function DesignChat({
  projectId,
  focusedImageId,
  focusedImageLabel,
  onClearFocus,
  onReadyForBrief,
  isGeneratingBrief = false,
  canGenerateBrief = false,
  initialMessage,
  conversationHistory,
  onHistoryUpdate,
}: DesignChatProps) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [suggestedActions, setSuggestedActions] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversationHistory]);

  const sendMessage = async (message: string) => {
    if (!message.trim() || isLoading || isGeneratingBrief) return;

    // Detect proceed intent — trigger brief generation directly, no AI round-trip
    if (canGenerateBrief && isProceedIntent(message)) {
      setInput("");
      onReadyForBrief();
      return;
    }

    const userMsg: ChatMessage = { role: "user", content: message, focused_image_id: focusedImageId ?? undefined };
    const updatedHistory = [...conversationHistory, userMsg];
    onHistoryUpdate(updatedHistory);
    setInput("");
    setIsLoading(true);
    setSuggestedActions([]);

    try {
      const response: ChatResponse = await chatWithProject(
        projectId,
        message,
        updatedHistory.slice(0, -1),
        focusedImageId ?? undefined,
      );

      const assistantMsg: ChatMessage = { role: "assistant", content: response.reply };
      onHistoryUpdate([...updatedHistory, assistantMsg]);
      setSuggestedActions(response.suggested_actions);

      if (response.ready_for_brief) {
        setSuggestedActions(["generate_brief"]);
      }
    } catch {
      const errorMsg: ChatMessage = {
        role: "assistant",
        content: "Sorry, I had trouble processing that. Could you try again?",
      };
      onHistoryUpdate([...updatedHistory, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChipSelect = (action: string) => {
    if (action === "generate_brief") {
      onReadyForBrief();
      return;
    }
    const chipMessage = action.replace(/_/g, " ");
    sendMessage(`I'd like to ${chipMessage}`);
  };

  const isDisabled = isLoading || isGeneratingBrief;
  const showBriefButton = canGenerateBrief && !isGeneratingBrief;
  const briefButtonProminent = suggestedActions.includes("generate_brief");

  return (
    <div className="flex flex-col h-full relative">
      {/* Brief generation loading overlay */}
      {isGeneratingBrief && (
        <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-b-lg">
          <div className="text-center space-y-3">
            <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto" />
            <div className="space-y-1">
              <p className="font-semibold text-sm">Creating your Design Brief...</p>
              <p className="text-xs text-muted-foreground">Analyzing the conversation to build your plan</p>
            </div>
          </div>
        </div>
      )}

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {initialMessage && conversationHistory.length === 0 && (
          <div className="flex gap-2">
            <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
              <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
            </div>
            <div className="bg-muted rounded-sm rounded-tl-none p-3 max-w-[85%]">
              <p className="text-sm whitespace-pre-wrap">{initialMessage}</p>
            </div>
          </div>
        )}

        {conversationHistory.map((msg, idx) => (
          <div key={idx} className={`flex gap-2 ${msg.role === "user" ? "justify-end" : ""}`}>
            {msg.role === "assistant" && (
              <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
              </div>
            )}
            <div className={`p-3 max-w-[80%] text-sm whitespace-pre-wrap ${
              msg.role === "user"
                ? "bg-secondary rounded-sm rounded-tr-none"
                : "bg-muted rounded-sm rounded-tl-none"
            }`}>
              {msg.content}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-2">
            <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
              <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
            </div>
            <div className="bg-muted rounded-sm rounded-tl-none p-3">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}

        {suggestedActions.length > 0 && !isLoading && !isGeneratingBrief && (
          <QuickReplyChips actions={suggestedActions} onSelect={handleChipSelect} />
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Footer: proceed button + input */}
      <div className="border-t">
        {/* Persistent "Generate Design Brief" button */}
        {showBriefButton && (
          <div className="px-3 pt-3">
            <Button
              onClick={onReadyForBrief}
              variant={briefButtonProminent ? "default" : "outline"}
              className={`w-full ${briefButtonProminent ? "animate-in fade-in slide-in-from-bottom-2 duration-300" : ""}`}
            >
              <FileText className="h-4 w-4 mr-2" />
              Generate Design Brief
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        )}

        <div className="p-3 space-y-2">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
              placeholder={
                isGeneratingBrief
                  ? "Generating design brief..."
                  : canGenerateBrief
                    ? 'Keep chatting, or say "go ahead" to generate the brief'
                    : "Describe what you'd like to visualize..."
              }
              className="flex-1 bg-muted border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
              disabled={isDisabled}
            />
            <Button size="sm" onClick={() => sendMessage(input)} disabled={!input.trim() || isDisabled}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
          {focusedImageId && (
            <div className="flex gap-2 items-center">
              <span className="text-[10px] text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                Focused on: {focusedImageLabel ?? focusedImageId}
              </span>
              <button onClick={onClearFocus} className="text-[10px] text-muted-foreground hover:text-foreground">
                × Clear
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

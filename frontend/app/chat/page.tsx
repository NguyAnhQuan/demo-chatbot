"use client";

import { useState, useEffect } from "react";

interface Message {
  sender: "user" | "bot";
  text: string;
}

interface Chat {
  id: number;
  title: string;
  messages: Message[];
}

export default function ChatPage() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [editingChatId, setEditingChatId] = useState<number | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const activeChat = chats.find((c) => c.id === activeChatId);

  // ✅ Khi load trang -> tự động tạo 1 cuộc trò chuyện
  useEffect(() => {
    if (chats.length === 0) {
      const newChat: Chat = {
        id: Date.now(),
        title: `Cuộc trò chuyện 1`,
        messages: [
          {
            sender: "bot",
            text: "Xin chào 👋! Tôi có thể giúp gì cho bạn hôm nay?",
          },
        ],
      };
      setChats([newChat]);
      setActiveChatId(newChat.id);
    }
  }, []);

  const startNewChat = () => {
    const newChat: Chat = {
      id: Date.now(),
      title: `Cuộc trò chuyện ${chats.length + 1}`,
      messages: [
        {
          sender: "bot",
          text: "Xin chào 👋! Tôi có thể giúp gì cho bạn hôm nay?",
        },
      ],
    };
    setChats((prev) => [...prev, newChat]);
    setActiveChatId(newChat.id);
  };

  const sendMessage = async () => {
    if (!input.trim() || !activeChatId) return;

    const userMsg: Message = { sender: "user", text: input };
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === activeChatId
          ? { ...chat, messages: [...chat.messages, userMsg] }
          : chat
      )
    );

    try {
      const res = await fetch("http://127.0.0.1:8000/backend/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      const data = await res.json();
      const botMsg: Message = { sender: "bot", text: data.reply };

      setChats((prev) =>
        prev.map((chat) =>
          chat.id === activeChatId
            ? { ...chat, messages: [...chat.messages, botMsg] }
            : chat
        )
      );
    } catch {
      const errorMsg: Message = {
        sender: "bot",
        text: "⚠️ Không thể kết nối đến server!",
      };
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === activeChatId
            ? { ...chat, messages: [...chat.messages, errorMsg] }
            : chat
        )
      );
    }

    setInput("");
  };

  const renameChat = (chatId: number) => {
    if (editTitle.trim()) {
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === chatId ? { ...chat, title: editTitle } : chat
        )
      );
    }
    setEditingChatId(null);
    setEditTitle("");
  };

  const deleteChat = () => {
    if (!activeChatId) return;
    const filteredChats = chats.filter((chat) => chat.id !== activeChatId);
    setChats(filteredChats);
    setActiveChatId(filteredChats[0]?.id || null);
  };

  return (
    <div className="flex h-screen bg-linear-to-br from-indigo-50 via-blue-50 to-purple-50 text-gray-900">
      {/* SIDEBAR */}
      <aside className="w-72 bg-white/80 backdrop-blur-sm border-r border-gray-200/50 flex flex-col shadow-lg">
        <div className="p-5 border-b border-gray-200/50 bg-linear-to-r from-blue-600 to-indigo-600 text-white">
          <h2 className="font-bold text-xl mb-3 flex items-center gap-2">
            💬 Lịch sử trò chuyện
          </h2>
          <button
            onClick={startNewChat}
            className="w-full bg-white/20 hover:bg-white/30 text-white px-4 py-2.5 rounded-xl text-sm font-medium shadow-md transition-all duration-200 hover:scale-105 flex items-center justify-center gap-2"
          >
            <span className="text-lg">+</span> Cuộc trò chuyện mới
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={`mb-2 rounded-xl transition-all duration-200 ${
                chat.id === activeChatId
                  ? "bg-linear-to-r from-blue-500 to-indigo-500 shadow-md"
                  : "bg-gray-50 hover:bg-gray-100"
              }`}
            >
              {editingChatId === chat.id ? (
                <div className="p-3 flex items-center gap-2">
                  <input
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") renameChat(chat.id);
                      if (e.key === "Escape") {
                        setEditingChatId(null);
                        setEditTitle("");
                      }
                    }}
                    className="flex-1 px-3 py-1.5 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 text-sm"
                    autoFocus
                  />
                  <button
                    onClick={() => renameChat(chat.id)}
                    className="text-green-600 hover:text-green-700 text-xl"
                    title="Lưu"
                  >
                    ✓
                  </button>
                  <button
                    onClick={() => {
                      setEditingChatId(null);
                      setEditTitle("");
                    }}
                    className="text-red-600 hover:text-red-700 text-xl"
                    title="Hủy"
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <div className="flex items-center">
                  <button
                    onClick={() => setActiveChatId(chat.id)}
                    className={`flex-1 text-left px-4 py-3 text-sm font-medium transition-colors ${
                      chat.id === activeChatId ? "text-white" : "text-gray-700"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-base">💭</span>
                      <span className="truncate">{chat.title}</span>
                    </div>
                  </button>
                  <button
                    onClick={() => {
                      setEditingChatId(chat.id);
                      setEditTitle(chat.title);
                    }}
                    className={`px-3 py-2 text-sm transition-opacity hover:opacity-100 ${
                      chat.id === activeChatId
                        ? "text-white/80"
                        : "text-gray-500 opacity-0 group-hover:opacity-100"
                    }`}
                    title="Đổi tên"
                  >
                    ✏️
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </aside>

      {/* MAIN CHAT AREA */}
      <div className="flex flex-col flex-1 bg-white/60 backdrop-blur-sm rounded-xl shadow-2xl overflow-hidden ml-2">
        {/* Header */}
        <header className="p-5 bg-linear-to-r from-blue-600 via-indigo-600 to-purple-600 text-white font-semibold text-lg flex justify-between items-center shadow-lg">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🤖</span>
            <span className="text-xl">
              {activeChat?.title || "Chọn cuộc trò chuyện"}
            </span>
          </div>
          {activeChat && (
            <button
              onClick={deleteChat}
              className="bg-red-500/20 hover:bg-red-500/30 text-white px-4 py-2 rounded-lg text-sm transition-all duration-200 hover:scale-105 flex items-center gap-2 backdrop-blur-sm"
            >
              🗑️ Xóa
            </button>
          )}
        </header>

        {/* Message List */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-linear-to-b from-transparent to-blue-50/30">
          {activeChat?.messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${
                msg.sender === "user" ? "justify-end" : "justify-start"
              } animate-fadeIn`}
            >
              <div
                className={`px-5 py-3 rounded-2xl max-w-xs md:max-w-md lg:max-w-lg transition-all duration-200 hover:scale-[1.02] ${
                  msg.sender === "user"
                    ? "bg-linear-to-r from-blue-600 to-indigo-600 text-white rounded-br-sm shadow-lg"
                    : "bg-white border border-gray-200/50 text-gray-800 rounded-bl-sm shadow-md"
                }`}
              >
                <div className="whitespace-pre-wrap wrap-break-word">
                  {msg.text}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Input Area */}
        {activeChat && (
          <div className="p-5 border-t border-gray-200/50 bg-white/80 backdrop-blur-sm flex items-center gap-3 shadow-lg">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              placeholder="💬 Nhập tin nhắn của bạn..."
              className="flex-1 border border-gray-300 rounded-full px-5 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 shadow-sm hover:shadow-md"
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim()}
              className="bg-linear-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 text-white px-8 py-3 rounded-full shadow-lg transition-all duration-200 hover:scale-105 disabled:scale-100 disabled:cursor-not-allowed font-medium"
            >
              Gửi 📤
            </button>
          </div>
        )}
      </div>

      <style jsx global>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }

        ::-webkit-scrollbar-track {
          background: transparent;
        }

        ::-webkit-scrollbar-thumb {
          background: linear-gradient(to bottom, #cbd5e1, #94a3b8);
          border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(to bottom, #94a3b8, #64748b);
        }
      `}</style>
    </div>
  );
}

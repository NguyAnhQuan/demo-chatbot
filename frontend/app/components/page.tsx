"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  const linkClass = (path: string) =>
    pathname === path
      ? "text-blue-600 font-semibold"
      : "text-gray-600 hover:text-blue-500";

  return (
    <nav className="flex justify-between items-center p-4 shadow-md bg-white">
      <h1 className="text-2xl font-bold">Chatbot</h1>
      <div className="flex gap-6 justify-center">
        <Link href="/chat">Chat</Link>
        <Link href="/file" className={linkClass("/file")}>
          Quản lý file
        </Link>
        <Link href="/data" className={linkClass("/data")}>
          Quản lý dữ liệu
        </Link>
      </div>
    </nav>
  );
}

"use client";

import { useEffect, useState } from "react";

interface FileItem {
  name: string;
  url: string;
}

export default function FileManager() {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  const API = "http://127.0.0.1:8000/backend/files/";

  const fetchFiles = async () => {
    const res = await fetch(API);
    const data = await res.json();
    setFiles(data);
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    setLoading(true);
    await fetch(API + "upload/", {
      method: "POST",
      body: formData,
    });
    setLoading(false);
    setSelectedFile(null);
    fetchFiles();
  };

  const handleDelete = async (filename: string) => {
    await fetch(API + `delete/${filename}/`, { method: "DELETE" });
    fetchFiles();
  };

  return (
    <div className="max-w-3xl mx-auto p-8 bg-white shadow-lg rounded-lg mt-10">
      <h1 className="text-2xl font-semibold text-blue-600 mb-6">
        📁 Quản lý File
      </h1>

      {/* Upload form */}
      <form onSubmit={handleUpload} className="flex gap-3 mb-6">
        <input
          type="file"
          onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
          className="border border-gray-300 rounded-md px-3 py-2 w-full"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Đang tải..." : "Tải lên"}
        </button>
      </form>

      {/* File list */}
      {files.length === 0 ? (
        <p className="text-center text-gray-500">Chưa có file nào</p>
      ) : (
        <ul className="divide-y divide-gray-200">
          {files.map((file) => (
            <li
              key={file.name}
              className="flex justify-between items-center py-3"
            >
              <div>
                <p className="font-medium">{file.name}</p>
              </div>
              <div className="flex gap-3">
                <a
                  href={`http://127.0.0.1:8000${file.url}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:underline"
                >
                  Xem
                </a>
                <button
                  onClick={() => handleDelete(file.name)}
                  className="text-red-500 hover:text-red-700"
                >
                  Xóa
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

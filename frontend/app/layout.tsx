import "./globals.css";
import Navbar from "./components/page";

export const metadata = {
  title: "Next.js + Django App",
  description: "Demo navigation dùng chung giữa các trang",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi">
      <body className="bg-gray-50 text-gray-900">
        <Navbar />
        <main className="p-6">{children}</main>
        <footer className="p-4 text-center text-gray-500 text-sm bg-gray-100 border-t">
          © {new Date().getFullYear()} MySite — Powered by Next.js & Django
        </footer>
      </body>
    </html>
  );
}

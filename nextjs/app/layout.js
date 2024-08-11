import Link from "next/link";

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <nav>
          <Link href="/">Home</Link>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}

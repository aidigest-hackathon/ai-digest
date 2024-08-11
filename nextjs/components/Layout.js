import Link from "next/link";

export default function Layout({ children }) {
  return (
    <div>
      <nav>
        <Link href="/">
          <a>Home</a>
        </Link>
      </nav>
      <main>{children}</main>
    </div>
  );
}

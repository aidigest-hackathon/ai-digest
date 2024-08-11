import Link from "next/link";

export default function ArticleList({ articles }) {
  return (
    <ul>
      {articles.map((article) => (
        <li key={article.id}>
          <Link href={`/article/${article.id}`}>{article.title}</Link>
        </li>
      ))}
    </ul>
  );
}

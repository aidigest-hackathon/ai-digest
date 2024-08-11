import { articles } from "../../../data/articles";

export default function Article({ params }) {
  const article = articles.find((a) => a.id === parseInt(params.id));

  if (!article) return <div>Article not found</div>;

  return (
    <>
      <h1>{article.title}</h1>
      <p>{article.content}</p>
    </>
  );
}

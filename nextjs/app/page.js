import ArticleList from "../components/ArticleList";
import { articles } from "../data/articles";

export default function Home() {
  return (
    <>
      <h1>Newspaper Articles</h1>
      <ArticleList articles={articles} />
    </>
  );
}

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Thesis}[2021/04/20 Bachelor Thesis Template]
\LoadClass{book}

% --- Margins ----
\usepackage{vmargin}
\setmarginsrb	{ 1.50in }  % left margin
				{  0.6in }  % top margin
				{ 1.00in }  % right margin
				{  0.8in }  % bottom margin
				{   20pt }  % head height
				{ 0.25in }  % head sep
				{    9pt }  % foot height
				{  0.3in }  % foot sep
				
\raggedbottom
\setlength{\topskip}{1\topskip \@plus 5\p@}
\doublehyphendemerits=10000       % No consecutive line hyphens.
\brokenpenalty=10000              % No broken words across columns/pages.
\widowpenalty=9999                % Almost no widows at bottom of page.
\clubpenalty=9999                 % Almost no orphans at top of page.
\interfootnotelinepenalty=9999    % Almost never break footnotes.
\usepackage{fancyhdr}
\lhead[\rm\thepage]{\fancyplain{}{\sl{\rightmark}}}
\rhead[\fancyplain{}{\sl{\leftmark}}]{\rm\thepage}
\chead{}\lfoot{}\rfoot{}\cfoot{}
\pagestyle{fancy}

% ---- University - Faculty - Department ----
\newcommand{\university}[3]{
	\vspace*{50px}
	\centering{
	\Large\textbf{#1} \par
	\Large\textbf{#2} \par
	\Large\textbf{#3}}
}

% ---- Thesis Type ----
\newcommand{\thesis}[1]{
	\bigskip\bigskip\bigskip
	\bigskip\bigskip\bigskip
	\bigskip\bigskip
	\centering{\Large #1}
}

% ---- Thesis Title ----
\newcommand{\topic}[1]{
	\bigskip
	\centering{\Huge\textbf{#1}}
}

% ---- Author ----
\newcommand{\authors}[1]{
	\bigskip\bigskip\bigskip
	\bigskip\bigskip\bigskip
	\bigskip\bigskip\bigskip
	\large submitted by \par
	\LARGE #1 \par
}

% ---- Hand In Date ----
\newcommand{\handin}[1]{
	\large on \par
	\large #1
}

% ---- Reviewers ----
\newcommand{\reviewers}[2]{
	\bigskip\bigskip\bigskip\bigskip\bigskip
	\large Reviewers \par
	\Large #1 \par
	\Large #2
}

% ---- Abstract ----
\newenvironment{abstract}{
	\setstretch{1.1}
	\begin{center}
		\huge{\textit{Abstract}} \par
		\bigskip\bigskip\bigskip\bigskip
	\end{center}
}

% ---- Acknowledgement ----
\newenvironment{acknowledgment}{
	\setstretch{1.1}
	\begin{center}
		\huge{\textit{Acknowledgment}} \par
		\bigskip\bigskip\bigskip\bigskip
	\end{center}
}

% ---- References ----
\newenvironment{references}{
	\setstretch{1.1}
	\begin{center}
		\huge{\textit{References}} \par
		\bigskip\bigskip\bigskip\bigskip
	\end{center}
}



























\newcommand{\blankpage}{
	\clearpage
	\null
	\thispagestyle{empty}
	\addtocounter{page}{-1}
	\clearpage
}

\newcommand{\para}{
	\vspace{1.5em} 
	\noindent
}








const board = document.getElementById("board");
const size = 15;
let currentPlayer = 1;
let Player2Color = {1: 'black', 2: 'white'};

const boardArray = Array(size).fill(0).map(() => Array(size).fill(0));

for (let i = 0; i < size * size; i++) {
    const cell = document.createElement('div');
    cell.classList.add('cell');
    cell.dataset.index = i;
    board.appendChild(cell);
}

board.addEventListener('click', (event) => {
    const cell = event.target;
    if (cell.classList.contains('cell') && !cell.innerHTML) {
        const index = +cell.dataset.index;
        const x = index % size;
        const y = Math.floor(index / size);

        const disc = document.createElement('div');
        disc.style.width = '30px';
        disc.style.height = '30px';
        disc.style.borderRadius = '50%';
        disc.style.backgroundColor = Player2Color[currentPlayer];
        cell.appendChild(disc);

        boardArray[y][x] = currentPlayer;

        console.log(boardArray);
        currentPlayer = currentPlayer === 1 ? 2 : 1;
}});